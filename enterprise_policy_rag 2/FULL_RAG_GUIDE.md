# Full local RAG upgrade

1. Install Ollama from https://ollama.com/download and launch it.
2. Run `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`.
3. In this folder, create/activate a Python 3.12 virtual environment and run `pip install -r requirements.txt`.
4. Run `streamlit run app_full.py` for the upgraded chatbot.
5. Run `python evaluate_full.py` for 15-question full-RAG results.
6. Inspect `evaluation_results_full.csv` and update the Word evaluation report with these **actual** results before submitting. The original Word report is for the TF-IDF baseline only.

The semantic index uses Ollama neural embeddings, saved locally as a JSON cache; retrieval uses cosine similarity via normalized NumPy vectors. The local Llama model receives only retrieved excerpts and is instructed to abstain if unsupported. This is a compact vector index, not Pinecone/Chroma. All policy documents are fictional. LLM answers and citations must be reviewed for accuracy. The `citations` list shows retrieved context, not a guarantee each source supports the final answer.
