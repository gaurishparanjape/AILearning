import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

chroma_client = chromadb.PersistentClient(path="./my_vector_db")

sentence_transformers_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(name="my_documents",
    embedding_function=sentence_transformers_ef,
    metadata={"hnsw:space": "cosine"})

with open("data.txt", "r") as f:
  raw_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
chunks = text_splitter.split_text(raw_text)

document_list = []
ids_list = []
metadatas_list = []

for i, chunk in enumerate(chunks):
  document_list.append(chunk)
  ids_list.append(f"id_{i}")
  metadatas_list.append({"source":"data.txt", "chunk_no":i})

  collection.add(documents=document_list, ids=ids_list, metadatas= metadatas_list)

print(f"Successful! Total {len(chunks)} chunks were saved in VectorDB")
