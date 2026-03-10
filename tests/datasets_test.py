from tqdm.auto import tqdm
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("myminimind/config/tokenizer")

total_tokens = 0

with open("dataset/sft_512.jsonl", "r") as f:
    for line in tqdm(f):
        conversations = json.loads(line)["conversations"]
        prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(prompt, add_special_tokens=False).input_ids
        total_tokens += len(tokens)

print(total_tokens)