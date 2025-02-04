
from flask import Flask, render_template, request, jsonify
from flask_session import Session
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['SESSION_TYPE'] = 'filesystem'  # Use the filesystem to store session data
app.config["SESSION_PERMANENT"] = True  # Sessions expire when the browser is closed
app.config["SESSION_USE_SIGNER"] = True  # Sign session cookies for added security
Session(app)

# MongoDB connection
MONGO_CONN = "<MONGODB_CONNECTION_STRING>"
client = MongoClient(MONGO_CONN, tls=True, tlsAllowInvalidCertificates=True)

# Define collections
faq_collection = client["document_search"]["documents"]

# Initialize Granite Embedding Model
model_path = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(model_path)

def generate_embedding(text):
    return embedding_model.encode(text).tolist()

# -----------------------------
# Unified Retrieval Function
# -----------------------------

# Function to retrieve HR knowledge from MongoDB
def retrieve_hr_knowledge(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    print("Fetching similar search for query: " + query)
    # MongoDB Atlas Search or Vector Search query

    pipeline = [
        {
            "$search": {
                "index": "default",  # Use the dynamic index name
                "knnBeta": {
                    "vector": query_embedding,  # Pass the query embedding here
                    "path": "embedding",  # Field where embeddings are stored
                    "k": top_k  # Number of nearest neighbors to retrieve
                }
            }
        },
        {
            "$project": {
                "content": 1,
                "metadata": 1,
                "score": {"$meta": "searchScore"}  # Include similarity score in results
            }
        }
    ]
    results = list(faq_collection.aggregate(pipeline))

    retrieved_docs = [doc["content"] for doc in results]
    metadata = [doc["metadata"] for doc in results]

    return retrieved_docs, metadata

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map="cpu")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").to("cpu")


def api_query(query):
    retrieved_knowledge, metadata  = retrieve_hr_knowledge(query)
    prompt_template = """
Here is your revised prompt tailored for a Product Recommendation Chatbot:

Role & Objective:

You are a Product Recommendation AI designed to assist customers in finding the best products based on their preferences, needs, and purchase history. Your responses must be personalized, well-structured, and aligned with available product data.

Read the context, which is in the form of JSON (containing product catalog, user preferences, and past interactions) along with the user query. Correlate both to generate a relevant and effective response as per the guidelines below:

Response Requirements:
	1.	Personalized Recommendations → Use retrieved product data and user preferences to suggest the most relevant items.
	2.	Structured Answer → Format the response clearly (e.g., bullet points, feature comparisons).
	3.	Reasoning & Justification → Explain why a particular product is recommended based on retrieved data.
	4.	User-Friendly Language → Ensure responses are clear, engaging, and easy to understand.
	5.	Actionable Steps → Provide guidance on how the user can proceed (e.g., add to cart, explore alternatives, check availability).

Retrieved Product Data:
{retrieved_knowledge}

Metadata:
{metadata}

User Query:
{question}

Output:
Generate a concise, clear, and human-readable product recommendation based on the above instructions. Ensure the response is informative and avoids unnecessary formatting or artifacts.
"""
    prompt = prompt_template.format(retrieved_knowledge=retrieved_knowledge, metadata=metadata, question=query)
    print(prompt)

    # Generate response using DeepSeek
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        response = model.generate(**input_ids, max_new_tokens=1000)
        print(response)
        print(tokenizer.decode(response[0], skip_special_tokens=False))
        return tokenizer.decode(response[0], skip_special_tokens=False)
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return jsonify({"error": "Failed to generate a response. Please try again."}), 500

@app.route("/")
def home():
    return render_template("index.html")  # Serves the chatbot UI

# API Endpoint for Chatbot to Get Response
@app.route("/api/", methods=["GET"])
def hr_chat():
    user_query = request.args.get("query", "")
    if not user_query:
        return jsonify({"response": "Please enter a valid question."})
    ai_response = api_query(user_query)
    print(ai_response)
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True)
