"""Pretrain a compact decoder, freeze its weights, then train low-rank factors."""
import json
import numpy as np
from app.core import ROOT, GROUPS, tokenize, softmax

def contexts(sentences, vocab):
    ids = {word:i for i,word in enumerate(vocab)}
    x, y = [], []
    for group, words in sentences:
        history = ["<s>", "<s>"] + words + ["</s>"]
        for a,b,c in zip(history, history[1:], history[2:]):
            x.append([GROUPS.index(group), ids[a], ids[b]])
            y.append(ids[c])
    return np.array(x), np.array(y)

def hidden(x, embeddings):
    return np.concatenate([table[x[:,i]] for i,table in enumerate(embeddings)], axis=1)

def train():
    base = []
    for line in (ROOT/"examples/base.txt").read_text().splitlines():
        category, sentence = line.split(" | ")
        base.append((category, tokenize(sentence)))
    marketing = []
    for line in (ROOT/"examples/pairs.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["split"] == "train":
            text = row["description"].replace(row["name"], "{name}").replace(" and ".join(row["facts"]), "{facts}")
            marketing.append((row["category"], tokenize(text)))
    vocab = sorted({w for _,ws in base+marketing for w in ws} | {"<s>", "</s>"})
    rng = np.random.default_rng(9)
    size = len(vocab)
    tables = [rng.normal(0,.35,(n,32)) for n in (3,size,size)]
    w = rng.normal(0,.03,(96,size))
    def optimize(sentences, params, gradients, steps):
        x,y = contexts(sentences,vocab)
        m,v = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        for t in range(1,steps+1):
            h = hidden(x,tables)
            matrix = w if len(params)==4 else w + params[0] @ params[1]
            probs = softmax(h @ matrix)
            if t in (1,steps):
                print(t, "loss", round(float(-np.log(probs[np.arange(len(y)),y]).mean()),4))
            probs[np.arange(len(y)),y] -= 1
            grads = gradients(h,probs/len(y),x)
            for i,(p,g) in enumerate(zip(params,grads)):
                m[i] = .9*m[i]+.1*g
                v[i] = .999*v[i]+.001*g*g
                p -= .015*(m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)
    def base_grad(h,d,x):
        parts = np.split(d @ w.T,3,axis=1)
        grads = [np.zeros_like(table) for table in tables]
        for i,g in enumerate(grads):
            np.add.at(g,x[:,i],parts[i])
        return grads+[h.T@d]
    optimize(base,[*tables,w],base_grad,180)
    out = ROOT/"models"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/"base.npz", vocab=vocab, group=tables[0], prev=tables[1], last=tables[2], weight=w)
    a,b = rng.normal(0,.02,(96,12)),np.zeros((12,size))
    def adapter_grad(h,d,x):
        return [h.T @ d @ b.T, (h@a).T @ d]
    optimize(marketing,[a,b],adapter_grad,700)
    np.savez_compressed(out/"adapter.npz", a=a,b=b)

if __name__ == "__main__":
    train()
