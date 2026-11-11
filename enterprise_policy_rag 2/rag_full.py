"""Local semantic retrieval and grounded answer generation using Ollama."""
import json
from pathlib import Path
import numpy as np
import requests
from rag import load_chunks

ROOT = Path(__file__).parent
URL = 'http://localhost:11434/api'
EMBED_MODEL = 'nomic-embed-text'
CHAT_MODEL = 'llama3.2:3b'

class FullPolicyRAG:
    def __init__(self):
        self.chunks = load_chunks()
        self.index_path = ROOT / '.embedding_cache.json'
        signature = [{'id': c['id'], 'text': c['text'], 'section': c['section']} for c in self.chunks]
        if self.index_path.exists():
            cache = json.loads(self.index_path.read_text())
        else:
            cache = {}
        if cache.get('signature') == signature and cache.get('model') == EMBED_MODEL:
            vectors = cache['vectors']
        else:
            texts = [f"{c['document']} | {c['section']} | {c['text']}" for c in self.chunks]
            vectors = self.embed(texts)
            self.index_path.write_text(json.dumps({'signature': signature, 'model': EMBED_MODEL, 'vectors': vectors}))
        self.matrix = np.asarray(vectors, dtype=np.float32)
        self.matrix /= np.maximum(np.linalg.norm(self.matrix, axis=1, keepdims=True), 1e-10)

    @staticmethod
    def embed(texts):
        response = requests.post(f'{URL}/embed', json={'model': EMBED_MODEL, 'input': texts}, timeout=180)
        response.raise_for_status()
        return response.json()['embeddings']

    def ask(self, question, top_k=4):
        if not question.strip():
            return {'answer': 'Please enter a question.', 'citations': [], 'confidence': 0, 'status': 'empty'}
        query = np.asarray(self.embed([question])[0], dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-10)
        scores = self.matrix @ query
        order = np.argsort(-scores)[:top_k]
        hits = [{**self.chunks[int(i)], 'score': round(float(scores[i]), 3)} for i in order]
        evidence = '\n\n'.join(f"[SOURCE {j}] {h['document']} / {h['section']} ({h['source']}): {h['text']}" for j, h in enumerate(hits, 1))
        prompt = ("You answer questions using ONLY the policy excerpts below. "
                  "If the excerpts do not explicitly contain enough information, say exactly: "
                  "'I could not find that information in the provided policies.' "
                  "Do not guess or use outside knowledge. Cite supporting excerpts using [SOURCE 1], [SOURCE 2], etc. "
                  "If the question is ambiguous, say what is unclear.\n\n"
                  f"POLICY EXCERPTS:\n{evidence}\n\nQUESTION: {question}")
        response = requests.post(f'{URL}/chat', json={'model': CHAT_MODEL, 'stream': False,
            'messages': [{'role': 'user', 'content': prompt}], 'options': {'temperature': 0}}, timeout=240)
        response.raise_for_status()
        answer = response.json()['message']['content'].strip()
        abstained = 'i could not find that information in the provided policies' in answer.lower()
        citations = [] if abstained else [{'document': h['document'], 'section': h['section'], 'source': h['source'], 'score': h['score']} for h in hits]
        return {'answer': answer, 'citations': citations, 'confidence': hits[0]['score'], 'status': 'abstained' if abstained else 'answered'}
