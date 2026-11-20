"""Produce inference weights once. No optimizer or training data required."""
import argparse
import numpy as np
from app.core import ROOT

def compact():
    out = ROOT/"models"
    with np.load(out/"base.npz") as base, np.load(out/"adapter.npz") as adapter:
        arrays = dict(base)
        arrays["weight"] = base["weight"] + adapter["a"] @ adapter["b"]
        np.savez_compressed(out/"merged.npz",**arrays)

def gpt2(base):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base)
    tuned = PeftModel.from_pretrained(model, ROOT/"models/gpt2-adapter").eval()
    output = ROOT/"models/gpt2-merged"
    tuned.merge_and_unload().save_pretrained(output)
    AutoTokenizer.from_pretrained(base).save_pretrained(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2", action="store_true")
    parser.add_argument("--base", default="distilgpt2")
    args = parser.parse_args()
    if args.gpt2:
        gpt2(args.base)
    else:
        compact()
