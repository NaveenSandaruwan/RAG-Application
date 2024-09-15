import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import faiss

# Load environment variables
load_dotenv()

# Ensure NVIDIA_API_KEY is set in the environment
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    raise ValueError("NVIDIA_API_KEY is not set in the environment variables.")

# Initialize the NVIDIA Embeddings model
embedder = NVIDIAEmbeddings()

# Define the document
document = "Artificial Intelligence (AI) is the simulation of human intelligence in machines."

# Split the document into chunks (simplified)
chunks = [document]

# Generate embeddings for each chunk
try:
    document_embeddings = embedder.embed_documents(chunks)
except Exception as e:
    print(f"Error generating document embeddings: {e}")

# Set up FAISS index for similarity search
dimension = len(document_embeddings[0]) if document_embeddings else 0  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension) if dimension > 0 else None
if index:
    index.add(np.array(document_embeddings))

# Set up the RAG pipeline
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer based on the following context:\n<Documents>\n{context}\n</Documents>"),
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
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        # Print the query to debug if necessary
        print(f"Generating response for query: {query}")

        # Directly invoke the chain with the query
        response = chain.invoke(query)

        # Print response details for debugging
        print(f"Response details: {response}")

        return response
    except Exception as e:
        # Print full error details for better diagnostics
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating a response."

# Test the RAG pipeline
if __name__ == "__main__":
    user_question = "What is AI?"
    response = generate_response(user_question)

    print("Generated answer:\n", response)
