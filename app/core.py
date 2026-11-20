"""Small conditional neural language model. The category and preceding token
predict the next word. Name and fact slots copy the submitted specifications."""
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("home", "travel", "desk")

def tokenize(text):
    return re.findall(r"\{\w+\}|[\w'-]+|[^\w\s]", text)

def softmax(x):
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)

def render(words, name, facts):
    line = re.sub(r"\s+([.,:!?])", r"\1", " ".join(words))
    return line.replace("{name}", name).replace("{facts}", facts)

class Writer:
    def __init__(self, mode="merged"):
        file = "merged.npz" if mode == "merged" else "base.npz"
        with np.load(ROOT/"models"/file) as base:
            self.vocab = base["vocab"].tolist()
            self.tables = [base[k] for k in ("group","prev","last")]
            self.weight = base["weight"]
        if mode == "adapter":
            with np.load(ROOT/"models/adapter.npz") as adapter:
                self.weight += adapter["a"] @ adapter["b"]
        elif mode not in ("base","merged"):
            raise ValueError("Unknown model mode")

    def generate(self, name, category, facts):
        name = name.strip()
        facts = [s.strip() for s in facts if s.strip()]
        if not name or not facts:
            raise ValueError("Enter a product name and at least one feature.")
        ids = [self.vocab.index("<s>")]*2
        words = []
        for _ in range(70):
            indices = (GROUPS.index(category), *ids[-2:])
            h = np.concatenate([table[i] for table,i in zip(self.tables,indices)])
            token = int((h @ self.weight).argmax())
            word = self.vocab[token]
            if word == "</s>":
                break
            words.append(word)
            ids.append(token)
        features = ", ".join(facts[:-1])+" and "+facts[-1] if len(facts)>1 else facts[0]
        return render(words,name,features)
