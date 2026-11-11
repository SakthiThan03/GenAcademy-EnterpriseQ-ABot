# AsterWorks Enterprise Policy Q&A Bot

A runnable **local RAG-style baseline** for a fictional enterprise knowledge base. The system ingests six Markdown policy documents, chunks them by heading, creates TF-IDF sparse vector embeddings, retrieves relevant sections by cosine similarity, and composes extractive, citation-backed answers. **No LLM, Pinecone, LangChain, paid API, or external service is used**; do not describe this implementation as an LLM-powered generative RAG system.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
On Windows: `.venv\Scripts\activate`. Open the local URL printed by Streamlit.

## Evaluate
```bash
python evaluate.py
```
Outputs `evaluation_results.csv` and `evaluation_summary.json`. Source precision and recall use the documents cited in the **answer**, not all top-k candidates. Answer checks use exact phrase matching or abstention and are only proxies, not human-graded factual correctness. Questions with no expected source are scored as 1 only when no sources are cited.

## Architecture
`policies/*.md` → heading-based chunks → TF-IDF (unigram/bigram) → cosine retrieval → relevance threshold → sentence selection → cited extractive answer. Files: `rag.py` retrieval logic, `app.py` UI, `evaluate.py` reproducible test runner, `evaluation_cases.json` gold set.

## Limitations and next steps
The retrieval model is lexical, not a dense neural embedding. It may miss synonyms, retrieve plausible but incomplete evidence, conflate related policies, or answer subjective questions. Source similarity is not confidence calibration. For production, use enterprise access control, redaction, version tracking, human-reviewed gold labels, dense embeddings, hybrid search, reranking, and grounded LLM synthesis with abstention checks. Do not upload confidential corporate documents without authorization.

## Demo
See `DEMO_SCRIPT.md`. Record a 2–3 minute screen capture locally; no recording is included in this repository.

## License
Educational demonstration. All AsterWorks policies are fictional.
