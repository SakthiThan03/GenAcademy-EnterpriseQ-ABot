import streamlit as st
from rag_full import FullPolicyRAG

st.set_page_config(page_title='AsterWorks RAG Assistant', page_icon='📚', layout='wide')
st.title('📚 AsterWorks Policy Assistant — Full RAG')
st.caption('Fictional enterprise policies • Ollama embeddings + semantic search + local LLM')
@st.cache_resource
def load_engine():
    return FullPolicyRAG()
try:
    rag = load_engine()
except Exception as exc:
    st.error(f'Unable to initialize the knowledge base. Confirm Ollama is running and both models are installed. Details: {exc}')
    st.stop()
question = st.text_input('Ask a policy question', placeholder='How do I request PTO?')
if st.button('Ask', type='primary') and question.strip():
    with st.spinner('Retrieving evidence and generating an answer...'):
        try:
            result = rag.ask(question)
        except Exception as exc:
            st.error(f'Could not generate an answer: {exc}')
            st.stop()
    st.subheader('Answer')
    st.write(result['answer'])
    st.caption(f"Status: {result['status']} | Top semantic similarity: {result['confidence']:.3f} (not a probability)")
    if result['citations']:
        st.subheader('Retrieved sources')
        for i, c in enumerate(result['citations'], 1):
            st.write(f"[SOURCE {i}] **{c['document']}** — {c['section']} (`{c['source']}`, similarity {c['score']:.3f})")
st.sidebar.info(f'{len(rag.chunks)} policy chunks indexed from 6 fictional documents.')
