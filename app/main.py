"""Run with streamlit run app/main.py."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from app.core import GROUPS, ROOT, Writer

@st.cache_resource
def writer():
    return Writer()

@st.cache_resource
def gpt_writer():
    from transformers import pipeline
    return pipeline("text-generation", model=str(ROOT/"models/gpt2-merged"), device=-1)

st.set_page_config(page_title="Product Description Enhancer")
st.title("Product Description Enhancer")
options = ["Compact LoRA"]
if (ROOT/"models/gpt2-merged/config.json").exists():
    options.insert(0,"GPT-2 LoRA")
engine = st.selectbox("Model", options)
with st.form("product"):
    name = st.text_input("Product name", placeholder="Atlas Pack", max_chars=120)
    category = st.selectbox("Category", GROUPS)
    specs = st.text_area("Features and search terms", placeholder="18 L capacity\nFront pocket", help="Enter one feature per line.", max_chars=1200)
    submitted = st.form_submit_button("Generate description")
if submitted:
    facts = [s.strip() for s in specs.splitlines() if s.strip()]
    if not name.strip() or not facts:
        st.error("Enter a product name and at least one feature.")
    else:
        if engine == "Compact LoRA":
            copy = writer().generate(name,category,facts)
        else:
            from tools.gpt2 import prompt
            generator = gpt_writer()
            copy = generator(prompt(name,category,facts),max_new_tokens=65,do_sample=False,
                             return_full_text=False,pad_token_id=generator.tokenizer.eos_token_id)[0]["generated_text"].strip()
        st.subheader("Product description")
        st.text(copy)
