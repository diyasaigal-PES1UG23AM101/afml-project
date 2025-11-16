# app/app.py
import streamlit as st
from src.rag import rag_answer

st.set_page_config(page_title="Tulu RAG Demo")
st.title("Tulu RAG — Ask in English or Tulu")

q = st.text_input("Question", "")
lang = st.radio("Response language", ("English", "Tulu", "Both"))

if st.button("Ask") and q.strip():
    with st.spinner("Retrieving..."):
        lang_code = {"English":"en","Tulu":"tulu","Both":"both"}[lang]
        answer, passages = rag_answer(q, lang=lang_code, top_k=10, rerank=True)
    st.subheader("Answer")
    st.write(answer)
    st.subheader("Retrieved passages (top 5)")
    for i, (idx, score, txt) in enumerate(passages):
        st.markdown(f"**[{idx}]** (score: {score:.3f})")
        st.write(txt[:800])
