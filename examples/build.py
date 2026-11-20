"""Build original paired records and independent neutral pretraining text."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = {
    "home": [("Alder Tray", "oak", "raised rim"), ("Vale Lamp", "linen shade", "brass base"), ("Reed Basket", "woven cotton", "two handles"), ("Fern Vase", "stoneware", "matte finish")],
    "travel": [("Atlas Pack", "18 L", "front pocket"), ("Drift Pouch", "canvas", "zip closure"), ("Ridge Flask", "500 ml", "steel body"), ("Coast Tote", "cotton", "flat base")],
    "desk": [("Grid Notebook", "dot grid", "96 pages"), ("Orbit Stand", "aluminum", "folding hinge"), ("Slate Mat", "felt", "70 cm width"), ("Arc Organizer", "bamboo", "three compartments")],
}
COPY = {
    "home": ["Give your space a considered detail with {name}. Its defining features include {facts}.", "Bring {name} into your everyday setting. Explore the details: {facts}."],
    "travel": ["Plan your next outing with {name}. Check the essentials before you pack: {facts}.", "Meet {name} for days on the move. Here are the details for your packing list: {facts}."],
    "desk": ["Make room on your desk for {name}. A closer look reveals {facts}.", "Set up your workspace with {name}. Its details at a glance: {facts}."],
}
def build():
    rows = []
    neutral = []
    for group, items in GROUPS.items():
        for i, (name, a, b) in enumerate(items):
            facts = f"{a} and {b}"
            description = COPY[group][i % 2].format(name=name, facts=facts)
            rows.append(dict(category=group, name=name, facts=[a,b], description=description,
                             split="evaluation" if i == 3 else "train"))
        neutral.extend([f"{group} | The item is {{name}}. The specifications include {{facts}}.",
                        f"{group} | This listing identifies {{name}}. Its details are {{facts}}."])
    (ROOT / "pairs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (ROOT / "base.txt").write_text("\n".join(neutral)+"\n")
if __name__ == "__main__":
    build()
