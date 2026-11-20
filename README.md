# Product Description Enhancer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/mddscymw9y/badge.svg)](https://ask.thili.ai/b/mddscymw9y)

This project provides e-commerce managers with a powerful tool to enhance their online storefronts. By fine-tuning a language model with LoRA, it transforms raw product specifications into engaging, SEO-optimized marketing descriptions. The solution leverages the `lfllm-lflora` technique to adapt a pre-trained LLM, demonstrating its ability to generate high-quality product copy.

## What you are building

**Goal.** You will build a tool that uses LoRA to fine-tune a language model, transforming basic product specifications into compelling, SEO-optimized marketing descriptions for e-commerce.

**Done when.** The project is done when you have a functional Streamlit UI that accepts product specifications and generates enhanced descriptions, proving the LoRA fine-tuning technique successfully adapts the LLM to create relevant and creative marketing copy after training and merging the adapter.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Compile a small dataset of product specs and marketing descriptions.**  
      Hint: Compile and format your product spec and description pairs into a suitable dataset.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Implement LoRA adapter injection and training on a pre-trained LLM using the `lfllm-lflora` technique.
- [ ] **3. Develop a Streamlit UI to input product specs and generate enhanced descriptions.**  
      Hint: Develop a Streamlit UI to handle input of product specs and display the generated descriptions.
- [ ] **4. Assess the creativity and relevance of the generated descriptions.**  
      Hint: Qualitatively assess the generated descriptions for creativity, relevance, and adherence to marketing goals.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Implement LoRA adapter merging for efficient inference, utilizing the `lfllm-lflora` technique.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Add options in the UI for users to specify the desired tone (e.g., formal, casual) for the generated descriptions.
- Allow users to input specific keywords that the generated descriptions should aim to include.
- Implement a simple feedback mechanism (e.g., thumbs up/down) for generated descriptions to gather qualitative data.
