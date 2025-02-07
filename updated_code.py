import os
import faiss
import torch
import pypdf
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Load Sentence Transformer model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Llama-3.2 model
llm = Llama(model_path="Llama-3.2-3B-Instruct-Q4_0.gguf", n_ctx=2048)


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    pdf_reader = pypdf.PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text


# Load and process the PDF
pdf_text = extract_text_from_pdf("harrypotter.pdf")
text_chunks = pdf_text.split(". ")  # Simple sentence-level chunking

# Generate embeddings
embeddings = embed_model.encode(text_chunks, convert_to_numpy=True)

# Store embeddings in FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index for future use
faiss.write_index(index, "harrypotter_index.faiss")


def retrieve_context(query, top_k=3):
    """Retrieve relevant text chunks based on the query."""
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return "\n".join([text_chunks[i] for i in indices[0]])


def generate_response(query):
    """Generate a response using Llama-3.2 with retrieved context."""
    context = retrieve_context(query)
    prompt = f"""
    Context:
    {context}

    User Query: {query}

    Answer concisely based on the context above.
    """
    output = llm(prompt)
    return output["choices"][0]["text"].strip()


if __name__ == "__main__":
    print("Welcome to the Harry Potter Q&A system! Type your questions below.")
    print("Type 'exit' to quit the application.")
    while True:
        query = input("Question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        response = generate_response(query)
        print(f"Answer: {response}\n")
