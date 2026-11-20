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
