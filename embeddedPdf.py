import os
import pdfplumber
import numpy as np
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import faiss

# Load environment variables
load_dotenv()

# Ensure NVIDIA_API_KEY is set in the environment
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    raise ValueError("NVIDIA_API_KEY is not set in the environment variables.")

# Initialize the NVIDIA Embeddings model
embedder = NVIDIAEmbeddings(model="NV-Embed-QA")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Define the path to your PDF
pdf_path = "C:\\Users\\MSI20\\Downloads\\Project Description 4 (2).pdf"

# Extract text from the PDF
document = extract_text_from_pdf(pdf_path)

# Split the document into chunks (e.g., sentences)
chunks = [chunk.strip() for chunk in document.split('.') if chunk.strip()]

# Generate embeddings for each chunk
document_embeddings = embedder.embed_documents(chunks)

# Set up FAISS index for similarity search
dimension = len(document_embeddings[0])  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings))

# Set up the RAG pipeline
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using information solely based on the following context:\n<Documents>\n{context}\n</Documents>"),
        ("user", "{question}"),
    ]
)

# Initialize the ChatNVIDIA model
chat_model = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Create a vector store and retriever
vectorstore = FAISS.from_texts(chunks, embedding=embedder)
retriever = vectorstore.as_retriever()

# Define the chain for the RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

# Function to generate a response
def generate_response(query):
    try:
        # Ensure the query is a string and properly formatted
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        response = chain.invoke(query)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating a response."

# Test the RAG pipeline
if __name__ == "__main__":
    while(True):
        user_question = input("enter query -- ")
        response = generate_response(user_question)

        print("Generated answer:\n", response)
