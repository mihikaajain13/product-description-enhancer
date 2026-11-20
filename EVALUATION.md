# Held-out product review

Each category has one unseen product. The compact model generates a sequence of words and slots. Slot substitution preserves the provided name and specifications. The reference copy for each held-out item was excluded from training.

| Product | Relevance and factuality | Creativity and marketing fit |
|---|---|---|
| Fern Vase | Includes stoneware and matte finish. | The opening places the vase in a living space. “Considered detail” is vague. |
| Coast Tote | Includes cotton and flat base. | The packing-list angle fits travel. It supplies no specific carrying benefit. |
| Arc Organizer | Includes bamboo and three compartments. | The desk opening fits the category. The second sentence reads like a catalog listing. |

All three frozen-base outputs use the same neutral opening. The adapted model produces three category-specific openings. It retains the searchable product names and six supplied features. The category-specific wording recurs across items in the same category. The sample measures neither organic search ranking nor sales performance.

Run `python -m tools.evaluate` to reproduce the comparison. Exact outputs are in `examples/evaluation.json`.

## DistilGPT-2 adapter

The pretrained and merged transformer received identical prompts and greedy decoding settings. On the three unseen products, adaptation replaced repetitive text with short marketing copy and retained all three product names. Fern Vase repeated “matte finish” and omitted stoneware. Coast Tote had a relevant travel opening but omitted both features. Arc Organizer had a generic opening and omitted both features. One of six supplied feature phrases appeared in the generated transformer copy. This small training set demonstrates a style change and limited specification coverage. Inspect generated copy before publishing. The compact decoder is more reliable for exact keyword coverage on this dataset.

The loaded merged checkpoint and the active LoRA adapter differed by at most 0.000031 in next-token logits on the merge probe. Exact before/after generations are in `examples/gpt2-evaluation.json`.
