"""Train LoRA on the public GPT-2 checkpoint; merge for standalone inference."""
import argparse
import json
from pathlib import Path
from app.core import ROOT

def prompt(name, category, facts):
    return f"Product: {name}\nCategory: {category}\nFeatures: {', '.join(facts)}\nDescription:"

def train(base="distilgpt2", epochs=8):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(9)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(base), LoraConfig(
        r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True,
        task_type="CAUSAL_LM"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.0005)
    rows = [json.loads(line) for line in (ROOT/"examples/pairs.jsonl").read_text().splitlines()]
    for _ in range(epochs):
        for row in rows:
            if row["split"] != "train":
                continue
            prefix = tokenizer.encode(prompt(row["name"],row["category"],row["facts"]))
            target = tokenizer.encode(" "+row["description"]) + [tokenizer.eos_token_id]
            inputs = torch.tensor([prefix+target],device=device)
            labels = inputs.clone()
            labels[:,:len(prefix)] = -100
            optimizer.zero_grad()
            model(input_ids=inputs,labels=labels).loss.backward()
            optimizer.step()
    model.peft_config["default"].base_model_name_or_path = base if not Path(base).is_dir() else "distilgpt2"
    model.save_pretrained(ROOT/"models/gpt2-adapter", save_embedding_layers=False)
    model.eval().merge_and_unload().save_pretrained(ROOT/"models/gpt2-merged")
    tokenizer.save_pretrained(ROOT/"models/gpt2-merged")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="distilgpt2")
    args = parser.parse_args()
    train(args.base)
