"""Compare frozen and adapted compact model outputs on unseen products."""
import json
from app.core import ROOT, Writer

def evaluate():
    base, tuned = Writer("base"), Writer("adapter")
    for line in (ROOT/"examples/pairs.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["split"] == "evaluation":
            args = row["name"],row["category"],row["facts"]
            yield {"name":args[0],"base":base.generate(*args),"adapted":tuned.generate(*args)}

if __name__ == "__main__":
    print(json.dumps(list(evaluate()),indent=2))
