"""Local extractive RAG: retrieval + grounded answer synthesis, no paid API required."""
from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).parent
STOP={'what','which','when','where','who','how','does','do','can','is','are','the','a','an','i','we','our','to','for','of','and','in','on','about','with','me','my','if','it','be','from','at','or','should','employees','employee','policy'}
def tokens(text): return {w for w in re.findall(r"[a-z0-9]+",text.lower()) if len(w)>2 and w not in STOP}
def load_chunks(folder=None):
    folder=Path(folder or ROOT/'policies'); chunks=[]
    for path in sorted(folder.glob('*.md')):
        lines=path.read_text(encoding='utf-8').splitlines(); title=lines[0].removeprefix('# ').strip(); section='Overview'; buf=[]
        def flush():
            if buf:
                body=' '.join(buf).strip()
                if body: chunks.append({'id':f'{path.stem}:{len(chunks)}','source':path.name,'document':title,'section':section,'text':body})
                buf.clear()
        for line in lines[1:]:
            if line.startswith('## '): flush(); section=line[3:].strip()
            elif line.strip() and not line.startswith('Document ID:'): buf.append(line.strip())
        flush()
    return chunks
class PolicyRAG:
    def __init__(self,folder=None):
        self.chunks=load_chunks(folder)
        if not self.chunks: raise ValueError('No policy chunks found')
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,stop_words='english')
        self.matrix=self.vectorizer.fit_transform([c['document']+' '+c['section']+' '+c['text'] for c in self.chunks])
    def ask(self,question,top_k=3,threshold=.09):
        if not question.strip(): return {'answer':'Please enter a question.','citations':[],'confidence':0,'status':'empty'}
        similarities=cosine_similarity(self.vectorizer.transform([question]),self.matrix)[0]
        order=np.argsort(-similarities)[:top_k]
        hits=[{**self.chunks[int(i)],'score':round(float(similarities[i]),3)} for i in order if similarities[i]>=threshold]
        if not hits: return {'answer':"I couldn't find enough evidence in the available policies to answer that. Please contact the relevant policy owner.",'citations':[],'confidence':round(float(max(similarities)),3),'status':'abstained'}
        qwords=tokens(question)
        candidates=[]
        for hit in hits:
            for sent in re.split(r'(?<=[.!?])\s+',hit['text']):
                overlap=len(qwords & tokens(sent))
                if overlap: candidates.append((overlap+hit['score'],sent,hit))
        candidates.sort(key=lambda x:-x[0]); selected=[]; used=set()
        for score,sent,hit in candidates:
            if sent not in used:
                selected.append((sent,hit));used.add(sent)
            if len(selected)==3:break
        if not selected: return {'answer':"I couldn't find enough evidence in the available policies to answer that.",'citations':[],'confidence':round(float(hits[0]['score']),3),'status':'abstained'}
        # Keep the answer entirely extractive to avoid hallucinated policy claims.
        return {'answer':' '.join(s for s,h in selected),'citations':[{'document':h['document'],'section':h['section'],'source':h['source'],'score':h['score']} for s,h in selected],'confidence':hits[0]['score'],'status':'answered'}
