import streamlit as st
from rag import PolicyRAG
st.set_page_config(page_title='AsterWorks Policy Assistant',page_icon='📚',layout='wide')
st.title('📚 AsterWorks Policy Assistant')
st.caption('Fictional company • Local, source-grounded retrieval • No external API key')
@st.cache_resource
def engine(): return PolicyRAG()
rag=engine()
st.info('This prototype uses TF-IDF retrieval and extractive answer synthesis, not an LLM. It is a lightweight RAG-style baseline; see README for limitations.')
question=st.text_input('Ask a policy question',placeholder='How many PTO days do full-time employees receive?')
if st.button('Find answer',type='primary') or question:
    if question.strip():
        result=rag.ask(question)
        st.subheader('Answer');st.write(result['answer'])
        st.caption(f"Status: {result['status']} | Top retrieval similarity: {result['confidence']:.3f} (not a calibrated probability)")
        if result['citations']:
            st.subheader('Source evidence')
            for i,c in enumerate(result['citations'],1): st.write(f"{i}. **{c['document']}** — {c['section']} (`{c['source']}`; similarity {c['score']:.3f})")
with st.sidebar:
    st.header('Knowledge base');st.write(f'{len(rag.chunks)} indexed policy sections from 6 fictional documents')
    st.markdown('**Example questions**')
    st.write('How do I request PTO?')
    st.write('What must remote employees do to stay secure?')
    st.write('What is the company stock price?')
    st.caption('Policies are fictional and for demonstration only.')
