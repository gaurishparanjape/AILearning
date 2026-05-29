import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

from docReadConVector import sentence_transformers_ef

st.set_page_config(layout="wide")
st.title("AI Semantic Search vs Traditional Keyword Search")

chroma_client = chromadb.PersistentClient(path="./my_vector_db")
sentence_transformers_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="my_documents", embedding_function=sentence_transformers_ef)


query = st.text_input("Example: How to cure headache")

if query:
  col1, col2 = st.columns(2)

  # --- डावा कॉलम: Semantic Search ---
  with col1:
    st.header("🤖 Semantic Search (Vector DB)")

    #Querying Database
    results = collection.query(query_texts=[query], n_results=2)

    if results and results['documents']:
      for doc, score, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
        similarity_score = round((1 - score) * 100, 2)
        with st.expander(f"Similarity Score: {similarity_score}% | Source: {meta['source']}"):
          st.write(doc)
    else:
      st.write("No result found in Database.")

  with col2:
    st.header("Keyword Search (Exact Match)")

    all_docs = collection.get()
    found = False

    if all_docs and all_docs['documents']:
      keywords = query.lower().split()
      for doc, meta in zip(all_docs['documents'], all_docs['metadatas']):
        if any(word in doc.lower() for word in keywords if len(word) > 3):
          found = True
          with st.expander(f"Match Found! | Source: {meta['source']}"):
            st.write(doc)

      if not found:
        st.warning("No keyword match found!")
    else:
      st.write("Databse is empty. Run ingest.py.")