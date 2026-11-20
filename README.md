# Product Description Enhancer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/mddscymw9y/badge.svg)](https://ask.thili.ai/b/mddscymw9y)

This project provides e-commerce managers with a powerful tool to enhance their online storefronts. By fine-tuning a language model with LoRA, it transforms raw product specifications into engaging, SEO-optimized marketing descriptions. The solution leverages the `lfllm-lflora` technique to adapt a pre-trained LLM, demonstrating its ability to generate high-quality product copy.

## What you are building

**Goal.** You will build a tool that uses LoRA to fine-tune a language model, transforming basic product specifications into compelling, SEO-optimized marketing descriptions for e-commerce.

**Done when.** The project is done when you have a functional Streamlit UI that accepts product specifications and generates enhanced descriptions, proving the LoRA fine-tuning technique successfully adapts the LLM to create relevant and creative marketing copy after training and merging the adapter.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Compile a small dataset of product specs and marketing descriptions.**  
      Hint: Compile and format your product spec and description pairs into a suitable dataset.
- [x] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Implement LoRA adapter injection and training on a pre-trained LLM using the `lfllm-lflora` technique.
- [x] **3. Develop a Streamlit UI to input product specs and generate enhanced descriptions.**  
      Hint: Develop a Streamlit UI to handle input of product specs and display the generated descriptions.
- [x] **4. Assess the creativity and relevance of the generated descriptions.**  
      Hint: Qualitatively assess the generated descriptions for creativity, relevance, and adherence to marketing goals.
- [x] **5. Implement adapter merging for inference.**  
      Hint: Implement LoRA adapter merging for efficient inference, utilizing the `lfllm-lflora` technique.

## Run the Streamlit app

From the repository folder with Python 3.12:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/main.py
```

Enter a product name, category and one feature or search term per line. Try **Coast Tote**, **travel**, **cotton** and **flat base**. The compact model runs from included merged NumPy weights, offline, without training or an API key.

## Enable the fine-tuned DistilGPT-2 model

The repository ships the trained rank-8 attention adapter. Download the public base model once and merge locally:

```bash
python -m pip install -r requirements-gpt2.txt
python -m tools.merge --gpt2
python -m streamlit run app/main.py
```

Select **GPT-2 LoRA** in the model selector. For an existing local DistilGPT-2 download, supply its directory with `--base`. The full merged transformer stays local to keep Git uploads small. The Dockerfile runs the compact mode by default.

## Training and evaluation

The 12 original product-specification pairs cover home, travel and desk products. Each category has three training products and one held-out product. The DistilGPT-2 path freezes the pretrained transformer and trains low-rank attention projection updates for eight epochs using loss on description tokens only. To retrain, use `python -m tools.gpt2`. Its adapter update is merged into the base weights for inference.

The compact alternative pretrains a conditional neural language model on neutral text and adapts its frozen output projection with rank-12 LoRA. It generates category-specific wording and copies supplied product names and features through slots. Rebuild with `python -m tools.train` then `python -m tools.merge`.

See EVALUATION.md for held-out outputs, feature coverage, creativity and limitations. Run `python -m pytest -q` for the quick form and merge checks. GitHub Actions runs those checks without training or downloading a transformer.
