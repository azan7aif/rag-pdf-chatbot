"""
Measure how good the RETRIEVAL step is (no LLM calls, so it's free).

1. Index your PDFs first (through the app, or python rag.py file.pdf "q").
2. Edit eval_questions.json with your own questions.
3. Run:  python evaluate.py

Each test case needs a question plus at least one of:
  - "expected_keyword": a phrase that must appear in the correct chunk
  - "expected_page": the page number where the answer lives
Optional: "source": the PDF file name.
"""
import json

import rag

K_VALUES = [1, 3, 5]


def is_hit(chunks, case):
    for c in chunks:
        if "source" in case and c["source"] != case["source"]:
            continue
        ok_page = "expected_page" not in case or c["page"] == case["expected_page"]
        ok_kw = (
            "expected_keyword" not in case
            or case["expected_keyword"].lower() in c["text"].lower()
        )
        if ok_page and ok_kw:
            return True
    return False


def main():
    with open("eval_questions.json", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Evaluating {len(cases)} questions\n")
    print(f"{'k':<5}{'hits':<8}{'hit rate'}")
    for k in K_VALUES:
        hits = sum(is_hit(rag.retrieve(c["question"], k=k), c) for c in cases)
        print(f"{k:<5}{hits}/{len(cases):<5}{hits / len(cases):.0%}")

    print("\nMissed questions at k=5:")
    for c in cases:
        if not is_hit(rag.retrieve(c["question"], k=5), c):
            print(" -", c["question"])


if __name__ == "__main__":
    main()
