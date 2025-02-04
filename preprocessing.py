import os
from pymongo import MongoClient, InsertOne
from langchain.document_loaders.json_loader import JSONLoader
from sentence_transformers import SentenceTransformer
import json

# -----------------------------
# Step 1: Load Amazon Product dataset
# -----------------------------
# Define dataset file path
dataset_file = './amazon-products.json'

if not os.path.isfile(dataset_file):
    raise FileNotFoundError(f"Dataset file not found at {dataset_file}. Please ensure the dataset exists.")
print(f"Dataset found at {dataset_file}")

# -----------------------------
# Step 2: Load Documents
# -----------------------------
# Load the dataset using LangChain's JSONLoader
print("Loading documents from dataset...")
loader = JSONLoader(file_path=dataset_file, jq_schema=".", text_content=False, json_lines=False)
docs = loader.load()
print(f"Loaded {len(docs)} documents successfully.")

# -----------------------------
# Step 3: Initialize Sentence Transformer Embedding Model
# -----------------------------
# Define embedding model details
model_path = "all-mpnet-base-v2"

# Load the Sentence Transformer model
print("Initializing Sentence Transformer Embedding Model...")
model = SentenceTransformer(model_path)
print("Embedding model initialized successfully.")

# -----------------------------
# Step 4: Initialize MongoDB Atlas
# -----------------------------
# Define MongoDB connection details
MONGO_CONN = "<MONGODB_CONNECTION_STRING>"

# Establish a MongoDB connection
print("Connecting to MongoDB Atlas...")
client = MongoClient(
    MONGO_CONN,
    tls=True,
    tlsAllowInvalidCertificates=True
)

# Define MongoDB collections for vector stores
collection = client["document_search"]["documents"]

# -----------------------------
# Step 5: Load Documents into Vector Store
# -----------------------------
# Define a helper function to process and store documents
def add_documents_to_vector_store(collection, data, key_field="text"):
    for item in data:
        try:
            content = item[key_field]  # Extract the key field for embeddings
            metadata = {k: v for k, v in item.items() if k != key_field}
            embedding = model.encode(content).tolist()  # Generate embeddings using SentenceTransformer
            # Store the document and its embedding in MongoDB
            collection.insert_one({
                "content": content,
                "embedding": embedding,
                "metadata": metadata
            })
        except Exception as e:
            print(f"Error processing item: {item} - Error: {e}")

# Load HR dataset
with open(dataset_file, "r") as file:
    dataset = json.load(file)

print("Loading product details into vector store...")
add_documents_to_vector_store(collection, dataset["faqs"], "text")  # Ensure "query" matches dataset field name
print("All documents successfully added to MongoDB Atlas with embeddings.")
