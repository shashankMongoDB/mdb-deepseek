
# Product Recommendation Assistant

A Flask-based web application that provides personalized product recommendations using a combination of vector-based document retrieval from **MongoDB Atlas** and response generation with a **DeepSeek large language model (LLM)**. The application leverages Sentence Transformers for embedding queries and product data, making it easy to search for relevant documents that inform the chatbot’s responses.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)


## Overview

This repository contains a product recommendation chatbot that:
- Retrieves relevant product data from a **MongoDB Atlas** using vector-based search.
- Generates personalized recommendations using a **DeepSeek LLM** (a distilled version of Qwen 1.5B).
- Provides a simple web interface and an API endpoint to interact with the chatbot.

The core idea is to embed both the user’s query and the stored product documents into a common vector space, retrieve the most similar documents, and then feed these as context to the LLM. The LLM then generates a clear, structured, and human-readable product recommendation based on the user’s query and retrieved data.

## Features
- **Flask Web Application**: Provides a user-friendly interface as well as an API endpoint.
- **Session Management**: Uses Flask-Session to manage user sessions securely.
- **MongoDB Integration**: Connects to **MongoDB Atlas** for storing and retrieving product data.
- **Vector Search**: Uses Sentence Transformers to convert text into embeddings, enabling semantic search via MongoDB’s Atlas Search (with k-NN).
- **DeepSeek LLM Integration**: Generates detailed product recommendations by using a custom prompt template to provide context and guidelines.
- **GPU/CUDA Support**: Optimized to run on GPU (CUDA) for faster inference with the LLM.

## Architecture
### Gen AI Architecture

<img width="1311" alt="Screenshot 2025-02-04 at 9 48 40 PM" src="https://github.com/user-attachments/assets/a253a4db-bfc6-422c-b6fe-25bea30eb072" />


### Sequence Flow & Information Architecture

<img width="1495" alt="Screenshot 2025-02-04 at 9 24 22 PM" src="https://github.com/user-attachments/assets/f303dec6-287b-44a8-abb9-96c46aad9cd2" />

### **Flow Description**
1. **User Interaction**: A user inputs a query via the web interface or API.
2. **Document Retrieval**:
   - The query is embedded using Sentence Transformers.
   - MongoDB Atlas Search is used to retrieve the top relevant product documents based on vector similarity.
3. **Prompt Formation**:
   - A prompt template is filled with the retrieved documents, metadata, and the user’s query.
4. **Response Generation**:
   - The prompt is fed into the DeepSeek LLM.
   - The model generates a response that is sliced to return only the newly generated tokens (excluding the prompt).
5. **Display Result**:
   - The generated recommendation is returned via the API and rendered in the web interface.

## Installation

### **Prerequisites**
- Python 3.8+
- CUDA-enabled GPU (Optional but recommended): The LLM is set to run on CUDA. If you don’t have a GPU, modify the code to run on CPU (this will be slower).
- MongoDB Atlas Account: For storing and accessing product data.
- Virtual Environment (Recommended): To manage dependencies without affecting your system Python.

### **Clone the Repository**
```sh
git clone https://github.com/yourusername/product-recommendation-chatbot.git
cd product-recommendation-chatbot
```

### **Install Dependencies**
It is recommended to use a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install the required Python packages:
```sh
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, ensure you install the following packages:
- Flask
- flask_session
- pymongo
- sentence-transformers
- transformers
- torch (and torchvision if needed)

## Configuration

### **MongoDB Connection**
The MongoDB connection string is hardcoded in the code. For production use, consider setting it as an environment variable:
```python
import os
MONGO_CONN = os.getenv("MONGO_CONN_STRING")
```

### **LLM and Embedding Model Setup**
- **DeepSeek LLM**: The code uses the DeepSeek model `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. Ensure you have the necessary access and GPU support.
- **Sentence Transformer**: The model `all-mpnet-base-v2` is used for generating embeddings. This model is available via Hugging Face’s sentence-transformers.

## Usage

### **Running the Application**
To start the Flask development server, run:
```sh
python main.py
```
By default, the application will run in debug mode on `http://127.0.0.1:5000`.

### **Interacting with the Chatbot**
- **Web Interface**:
  Navigate to `http://127.0.0.1:5000` to access the chatbot UI.
  <img width="1512" alt="Screenshot 2025-02-04 at 8 58 28 PM" src="https://github.com/user-attachments/assets/af843922-4c0b-42e0-a3d3-8634ca331073" />
  <img width="1511" alt="Screenshot 2025-02-04 at 8 59 07 PM" src="https://github.com/user-attachments/assets/be815b89-0881-48d9-8935-43a534e82a93" />


- **API Endpoint**:
  You can also query the chatbot directly via the API endpoint:
  ```sh
  curl "http://127.0.0.1:5000/api/?query=Your+question+here"
  ```

The API will return a JSON response with the generated product recommendation.

## Code Overview

### **Main Components**
- **Flask App Setup**: Configures the Flask application, session management, and routes.
- **MongoDB Integration**: Uses `pymongo` to connect to a MongoDB Atlas cluster.
- **Embedding and Retrieval**: Uses `SentenceTransformer` for query-document similarity search.
- **Prompt Template and LLM Generation**: Constructs a prompt, sends it to DeepSeek LLM, and extracts generated recommendations.
- **Flask Endpoints**:
  - `/`: Serves the chatbot user interface.
  - `/api/`: Accepts GET requests with a query parameter and returns the chatbot response in JSON format.

## Troubleshooting

### **CUDA Issues**
If you do not have a CUDA-enabled GPU, modify the model and tokenizer calls to use CPU by replacing `.to("cuda")` with `.to("cpu")`. Be aware that this will likely slow down the inference process.

### **MongoDB Connection Errors**
Ensure your MongoDB connection string is correct and that your IP is whitelisted in MongoDB Atlas.

### **Dependency Problems**
Make sure all dependencies are installed in your virtual environment. If you encounter version conflicts, consider creating a fresh environment and reinstalling.

**Note:** This implementation is a proof-of-concept demonstration rather than a fully developed, production-ready chatbot. It exemplifies how a product recommendation assistant can leverage the DeepSeek LLM to generate contextually reasoned product recommendations. Future enhancements could integrate advanced NLP functionalities such as intent classification, sentiment analysis, and additional conversational capabilities to create a more robust and comprehensive solution.

---

