from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load text
with open("harry_potter_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_text(text)

# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_texts(text_chunks, embeddings, persist_directory="chroma_db")

# Persist ChromaDB
vectorstore.persist()
print("ChromaDB populated successfully!")

