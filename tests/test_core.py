import json
import numpy as np
from streamlit.testing.v1 import AppTest
from app.core import ROOT, Writer

def test_merge_and_held_out():
    merged, adapter = Writer(), Writer("adapter")
    np.testing.assert_allclose(merged.weight, adapter.weight, atol=1e-10)
    rows = [json.loads(s) for s in (ROOT/"examples/pairs.jsonl").read_text().splitlines()]
    for row in rows:
        if row["split"] == "evaluation":
            args = row["name"], row["category"], row["facts"]
            text = merged.generate(*args)
            assert text == adapter.generate(*args)
            assert all(term in text for term in [row["name"], *row["facts"]])

def test_streamlit_form():
    page = AppTest.from_file(str(ROOT/"app/main.py")).run()
    page.selectbox[0].select("Compact LoRA")
    page.text_input[0].input("Desk Tray")
    page.text_area[0].input("walnut\nraised edges")
    page.button[0].click().run()
    assert not page.exception
    assert "Desk Tray" in page.text[0].value
