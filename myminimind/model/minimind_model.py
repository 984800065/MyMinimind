import time
import math
import torch
import torch.nn.functional as F

from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union

from myminimind.model.minimind_config import MiniMindConfig
from myminimind.utils.logger import logger

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps).type_as(x_fp32)
        return rms.to(x.dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self._rms(x) * self.weight.to(x.dtype)


class YaRN:
    def __init__(
        self,
        dim: int,
        beta_fast: int,
        beta_slow: int,
        factor: int,
        train_seq_len: int,
        base: float = 1e6,
    ):
        self.dim = dim
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.factor = factor
        self.train_seq_len = train_seq_len
        self.base = base

    def _beta_to_index(self, beta: int) -> int: 
        return (self.dim // 2) * (math.log(self.train_seq_len / (2 * math.pi * beta)) / math.log(self.base))
    
    def interpolate_freqs(self, freqs: torch.Tensor) -> torch.Tensor:
        low_index = max(self._beta_to_index(self.beta_fast), 0)
        high_index = min(self._beta_to_index(self.beta_slow), self.dim // 2 - 1)
        ramp = torch.clamp((torch.arange(self.dim // 2) - low_index) / max(high_index - low_index, 1e-3), 0, 1)

        interpolated_freqs = freqs * (1 - ramp + ramp / self.factor)
        return interpolated_freqs

    def __call__(self, freqs: torch.Tensor) -> torch.Tensor:
        return self.interpolate_freqs(freqs)


class RoPEWithInterpolation(nn.Module):
    def __init__(
        self,
        dim: int,
        now_seq_len: int, 
        params: dict,
        base: float = 1e6,
        inference_rope_scaling: bool = False,
    ):
        super().__init__()
        # (dim // 2, )
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))

        train_seq_len = params.get("train_seq_len", 2048)
        if now_seq_len > train_seq_len and inference_rope_scaling:
            interpolation_type = params.get("interpolation_type", "yarn")
            
            if interpolation_type == "yarn":
                factor = params.get("factor", 16)
                beta_fast = params.get("beta_fast", 32)
                beta_slow = params.get("beta_slow", 1)
                yarn = YaRN(dim, beta_fast, beta_slow, factor, train_seq_len, base)
                # (dim // 2, )
                freqs = yarn(freqs)
            else:
                raise ValueError(f"Invalid interpolation type: {interpolation_type}")
        
        # (now_seq_len, )
        t = torch.arange(now_seq_len, device=freqs.device)

        # (now_seq_len, dim // 2)
        phi = torch.outer(t, freqs)

        attention_factor = params.get("attention_factor", 1.0)
        # (now_seq_len, dim)
        cos_phi = torch.cat([phi.cos(), phi.cos()], dim=-1) * attention_factor
        self.register_buffer("cos_phi", cos_phi)
        # (now_seq_len, dim)
        sin_phi = torch.cat([phi.sin(), phi.sin()], dim=-1) * attention_factor
        self.register_buffer("sin_phi", sin_phi)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, num_heads, head_dim)
        return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, begin_pos: int, end_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # end_pos - begin_pos == seq_len
        # q.shape == (batch_size, seq_len, num_heads, head_dim)
        # k.shape == (batch_size, seq_len, num_key_value_heads, head_dim)
        # self.cos_phi.shape == self.sin_phi.shape == (now_seq_len, dim) == (seq_len, dim)

        # self.cos_phi[None, :, None, :].shape == (1, seq_len, 1, dim)
        # self.sin_phi[None, :, None, :].shape == (1, seq_len, 1, dim)

        # (batch_size, seq_len, num_heads, head_dim)
        q = q * self.cos_phi[None, begin_pos:end_pos, None, :] + self._rotate_half(q) * self.sin_phi[None, begin_pos:end_pos, None, :]
        # (batch_size, seq_len, num_key_value_heads, head_dim)
        k = k * self.cos_phi[None, begin_pos:end_pos, None, :] + self._rotate_half(k) * self.sin_phi[None, begin_pos:end_pos, None, :]

        return q, k

    def __call__(self, q: torch.Tensor, k: torch.Tensor, begin_pos: int, end_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rope(q, k, begin_pos, end_pos)


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        config: MiniMindConfig,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = self.hidden_size // self.num_heads

        self.group_num = config.group_num
        assert self.num_heads % self.group_num == 0, "num_heads must be divisible by group_num"
        self.num_key_value_heads = self.num_heads // self.group_num

        self.norm = RMSNorm(self.hidden_size, config.rms_norm_eps)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
        
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self.is_flash_attention = config.flash_attention

    def _repeat_kv(self, x: torch.Tensor, repeat_times: int) -> torch.Tensor:
        # return torch.repeat_interleave(x, repeat_times, dim=2)
        
        # x.shape == (batch_size, seq_len, num_key_value_heads, head_dim)
        batch_size, seq_len, num_key_value_heads, head_dim = x.shape
        if repeat_times == 1:
            return x
        else:
            return x[:, :, :, None, :].expand(
                batch_size, seq_len, num_key_value_heads, repeat_times, head_dim
            ).reshape(batch_size, seq_len, num_key_value_heads * repeat_times, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: RoPEWithInterpolation,
        begin_pos: int,
        end_pos: int,
        use_kv_cache: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # x.shape == (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = x.shape
        num_heads = self.num_heads
        num_key_value_heads = self.num_key_value_heads
        head_dim = self.head_dim

        x = self.norm(x)

        # (batch_size, seq_len, num_heads * head_dim)
        q: torch.Tensor = self.q_proj(x)
        # (batch_size, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size, seq_len, num_heads, head_dim)

        # (batch_size, seq_len, num_key_value_heads * head_dim)
        k: torch.Tensor = self.k_proj(x)
        # (batch_size, seq_len, num_key_value_heads, head_dim)
        k = k.reshape(batch_size, seq_len, num_key_value_heads, head_dim)

        # (batch_size, seq_len, num_key_value_heads * head_dim)
        v: torch.Tensor = self.v_proj(x)
        # (batch_size, seq_len, num_key_value_heads, head_dim)
        v = v.reshape(batch_size, seq_len, num_key_value_heads, head_dim)

        # (batch_size, seq_len, num_heads, head_dim), (batch_size, seq_len, num_key_value_heads, head_dim)
        q, k = position_embeddings(q, k, begin_pos, end_pos)

        if past_key_values is not None:
            # Attention, this implementation of KV cache has wrong time complexity which is still O(n ^ 2)
            # (batch_size, seq_len + past_seq_len, num_key_value_heads, head_dim)
            k = torch.cat([past_key_values[0], k], dim=1)
            # (batch_size, seq_len + past_seq_len, num_key_value_heads, head_dim)
            v = torch.cat([past_key_values[1], v], dim=1)

        past_kv = (k, v) if use_kv_cache else None
        
        # (batch_size, num_heads, seq_len, head_dim)
        q = q.permute(0, 2, 1, 3)

        assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
        repeat_times = num_heads // num_key_value_heads
        # (batch_size, num_heads, seq_len + past_seq_len, head_dim)
        k = self._repeat_kv(k, repeat_times).permute(0, 2, 1, 3)
        # (batch_size, num_heads, seq_len + past_seq_len, head_dim)
        v = self._repeat_kv(v, repeat_times).permute(0, 2, 1, 3)

        if self.is_flash_attention and seq_len > 1 and past_key_values is None and (attention_mask is None or torch.all(attention_mask == 1)):
            # (batch_size, num_heads, seq_len, head_dim)
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.config.dropout if self.training else 0.0, is_causal=True)
        else:
            # (batch_size, num_heads, seq_len, seq_len + past_seq_len)
            attention_scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(head_dim)
            # (batch_size, num_heads, seq_len, seq_len + past_seq_len)
            attention_scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=attention_scores.device), diagonal=1)

            # attention_mask.shape == (batch_size, seq_len + past_seq_len), full with 1 and 0. 1 means valid, 0 means masked.
            if attention_mask is not None:
                # (batch_size, 1, 1, seq_len + past_seq_len)
                extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -1e9
                # (batch_size, num_heads, seq_len, seq_len + past_seq_len)
                attention_scores = attention_scores + extended_attention_mask

            # (batch_size, num_heads, seq_len, seq_len + past_seq_len)
            attention_weights = F.softmax(attention_scores, dim=-1).type_as(q)
            attention_weights = self.attention_dropout(attention_weights)
            # (batch_size, num_heads, seq_len, head_dim)
            output = torch.einsum("bhij,bhjd->bhid", attention_weights, v)

        # (batch_size, seq_len, num_heads, head_dim)
        output = output.permute(0, 2, 1, 3)
        # (batch_size, seq_len, num_heads * head_dim) == (batch_size, seq_len, hidden_size)
        output = output.reshape(batch_size, seq_len, -1)
        # (batch_size, seq_len, hidden_size)
        output = self.residual_dropout(self.out_proj(output))
        return output, past_kv


class GLU_FFN(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.glu_ffn = GLU_FFN(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # multi return to compatible with MoEFeedForward
        return self.glu_ffn(self.norm(x)), x.new_tensor(0.)


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_routed_experts = config.num_routed_experts

        self.scoring_function = config.scoring_function
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob

        # (num_routed_experts, hidden_size)
        self.weight = nn.Parameter(torch.empty((self.num_routed_experts, config.hidden_size)))
        self.reset_parameter()

    def reset_parameter(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        # hidden_states.shape == (batch_size, seq_len, hidden_size)

        # (batch_size, seq_len, num_routed_experts)
        logits = F.linear(hidden_states, self.weight, None).float()
        logits = logits - logits.max(dim=-1, keepdim=True).values
        if self.scoring_function == "softmax":
            # (batch_size, seq_len, num_routed_experts)
            scores = logits.softmax(dim=-1).to(hidden_states.dtype)
        else:
            raise NotImplementedError(f"unsupportable scoring function for MoE gating: {self.scoring_function}")

        # (batch_size, seq_len, top_k), (batch_size, seq_len, top_k)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.training and self.alpha > 0.0:
            if self.seq_aux:
                # DeepSeek-v2 style auxiliary loss
                # (batch_size, num_routed_experts)
                p = scores.mean(dim=1)
                
                # (batch_size, seq_len, num_routed_experts)
                f = torch.zeros((batch_size, seq_len, self.num_routed_experts), device=hidden_states.device)
                src = torch.ones(topk_idx.shape, device=hidden_states.device, dtype=f.dtype)
                f.scatter_add_(dim=2, index=topk_idx, src=src)
                # (batch_size, seq_len, num_routed_experts)
                f = self.num_routed_experts / (self.top_k * seq_len) * f
                # (batch_size, num_routed_experts)
                f = f.mean(dim=1)

                # (batch_size, )
                aux_loss = self.alpha * (f * p).sum(dim=1)
                # ()
                aux_loss = aux_loss.mean()
            else:
                # Switch Transformer style auxiliary loss
                # (batch_size * seq_len, num_routed_experts)
                p = scores.reshape(-1, self.num_routed_experts)
                # (num_routed_experts, )
                p = p.mean(dim=0)

                # (batch_size, seq_len, num_routed_experts)
                f = torch.zeros((batch_size, seq_len, self.num_routed_experts), device=hidden_states.device)
                src = torch.ones(topk_idx.shape, device=hidden_states.device, dtype=f.dtype)
                f.scatter_add_(dim=2, index=topk_idx, src=src)
                # (batch_size * seq_len, num_routed_experts)
                f = f.reshape(-1, self.num_routed_experts)
                # (num_routed_experts, )
                f = f.mean(dim=0)

                # ()
                aux_loss = self.alpha * self.num_routed_experts * (f * p).sum()
        else:
            # ()
            aux_loss = scores.new_tensor(0.)
        
        return topk_idx, topk_weight, aux_loss


class MoEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([GLU_FFN(config) for _ in range(config.num_routed_experts)])
        self.gate = MoEGate(config)
        self.shared_experts = nn.ModuleList([GLU_FFN(config) for _ in range(config.num_shared_experts)])
        self.capacity_factor = config.capacity_factor
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = x.shape
        top_k = self.config.num_experts_per_token
        x = self.norm(x)

        # topk_idx.shape == (batch_size, seq_len, top_k)
        # topk_weights.shape == (batch_size, seq_len, top_k)
        # aux_loss.shape == ()
        topk_idx: torch.Tensor
        topk_weight: torch.Tensor
        aux_loss: torch.Tensor
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # avoid bugs
        topk_idx = topk_idx.contiguous()
        topk_weight = topk_weight.contiguous()

        expert_capacity = math.ceil(batch_size * seq_len * top_k / self.config.num_routed_experts * self.capacity_factor)

        # (batch_size * seq_len, hidden_size)
        flat_x = x.reshape(-1, hidden_size)
        # (batch_size * seq_len, hidden_size)
        flat_y = torch.zeros_like(flat_x)

        # (batch_size * seq_len * top_k, )
        flat_expert_indices = topk_idx.reshape(-1)

        if self.config.norm_topk_prob:
            # (batch_size, seq_len, top_k)
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            topk_weight = topk_weight.to(flat_x.dtype)
        # (batch_size * seq_len * top_k, )
        flat_expert_weights = topk_weight.reshape(-1)

        # only difference between new_token and token is that new_token // top_k == token
        # (batch_size * seq_len * top_k, )
        new_token_indices_sorted_by_expert_id = torch.argsort(flat_expert_indices)
        # (batch_size * seq_len * top_k, )
        token_indices_sorted_by_expert_id = new_token_indices_sorted_by_expert_id // top_k

        # (num_routed_experts, )
        num_token_per_expert = flat_expert_indices.bincount(minlength=self.config.num_routed_experts).cumsum(dim=0)

        start_index = 0
        # expert parallel simulation and all-to-all simulation
        for expert_index, expert in enumerate(self.experts):
            next_start_index = num_token_per_expert[expert_index].item()
            cur_index_slice = slice(
                start_index,
                min(start_index + expert_capacity, next_start_index)
            )
            start_index = next_start_index

            cur_selected_flat_new_token_indices = new_token_indices_sorted_by_expert_id[cur_index_slice]
            cur_selected_flat_token_indices = token_indices_sorted_by_expert_id[cur_index_slice]

            if self.training and cur_index_slice.stop == cur_index_slice.start:
                dummy = sum(p.sum() for p in expert.parameters()) * 0.0
                flat_y = flat_y + dummy
            else:
                # (cur_selected_token_num, hidden_size)
                tmp_flat_y = expert(flat_x[cur_selected_flat_token_indices])
                # (cur_selected_token_num, )
                tmp_flat_topk_weight = flat_expert_weights[cur_selected_flat_new_token_indices]
                # flat_y[cur_selected_flat_token_indices] += tmp_flat_y * tmp_flat_topk_weight[:, None]
                flat_y.index_add_(dim=0, index=cur_selected_flat_token_indices, source=tmp_flat_y * tmp_flat_topk_weight[:, None])

        # shared expert parallel simulation
        if len(self.shared_experts) > 0:
            scale = 1.0 / len(self.shared_experts)
            for shared_expert in self.shared_experts:
                flat_y += shared_expert(flat_x) * scale

        # (batch_size * seq_len, hidden_size)
        y = flat_y.reshape(batch_size, seq_len, hidden_size)

        self.aux_loss = aux_loss
        # original return function
        # return y

        # my return function
        return y, self.aux_loss


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attention = GroupQueryAttention(config)

        self.layer_id = layer_id
        self.mlp = FeedForward(config) if not config.use_moe else MoEFeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: RoPEWithInterpolation,
        begin_pos: int,
        end_pos: int,
        use_kv_cache: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        residual = hidden_states

        hidden_states, present_key_values = self.self_attention(
            hidden_states,
            position_embeddings,
            begin_pos,
            end_pos,
            use_kv_cache,
            past_key_values,
            attention_mask
        )

        hidden_states += residual

        residual = hidden_states
        hidden_states, aux_loss = self.mlp(hidden_states)
        hidden_states += residual

        return hidden_states, present_key_values, aux_loss


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([MiniMindBlock(layer_id, config) for layer_id in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        assert config.hidden_size % config.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        head_dim = config.hidden_size // config.num_attention_heads
        self.position_embeddings = RoPEWithInterpolation(
            dim=head_dim,
            now_seq_len=config.max_seq_len,
            params=config.rope_params,
            base=config.rope_base,
            inference_rope_scaling=config.inference_rope_scaling,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        # input_ids.shape == (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * self.num_hidden_layers

        # past_key_values[0][0].shape == (batch_size, past_seq_len, num_heads, head_dim)
        begin_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        end_pos = begin_pos + seq_len

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        aux_loss = torch.tensor(0., device=hidden_states.device)
        presents_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            layer: MiniMindBlock
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]
            hidden_states, present_key_values, layer_aux_loss = layer(
                hidden_states,
                self.position_embeddings,
                begin_pos=begin_pos,
                end_pos=end_pos,
                use_kv_cache=use_cache,
                past_key_values=past_key_value,
                attention_mask=attention_mask
            )
            presents_key_values.append(present_key_values)
            aux_loss += layer_aux_loss
        
        hidden_states: torch.Tensor = self.norm(hidden_states)

        return hidden_states, presents_key_values, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig):
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        hidden_states, presents_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # hidden_states.shape == (batch_size, seq_len, hidden_size)
        logits: torch.Tensor = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # label中句子完成之后的padding token的id被赋值成了-100，因此这些token不计入损失
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

            assert not math.isnan(loss), f"loss is nan, shift_logits: {shift_logits}, shift_labels: {shift_labels}"

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=presents_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss

        return output