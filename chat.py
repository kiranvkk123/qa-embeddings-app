import os
import openai
import pinecone
from flask import Flask, render_template, request
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize OpenAI API
openai.api_key = 'your open ai key'

# Initialize Pinecone client
pc = Pinecone(api_key="pinecone key", environment="us-east-1")

# Create the index if it doesn't exist
index_name = 'qa-index'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding size
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Function to get text embeddings using OpenAI API
def get_embeddings(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Use OpenAI embedding model
        input=text
    )
    embeddings = np.array(response['data'][0]['embedding'])
    return embeddings

# Preprocess your knowledge base (could be a list of data science-related documents)
documents = [
    "What is Data Science?",
    "What are the different types of machine learning?",
    "Explain supervised learning.",
    "What is a neural network?",
    # Add more documents related to your knowledge base
]

# Insert embeddings into Pinecone
def insert_embeddings():
    vectors = []
    for i, doc in enumerate(documents):
        embedding = get_embeddings(doc)
        vectors.append((f"doc_{i}", embedding.tolist()))  # Converting numpy array to list

    # Insert into Pinecone
    index.upsert(vectors=vectors)

# Insert embeddings into Pinecone only if index is empty (for example, on first run)
if not index.describe_index_stats()['total_vector_count']:
    insert_embeddings()

# Flask app setup
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    question = request.form.get("question")
    answer = ""

    if question:
        answer = ask_question(question)

    return render_template("index.html", question=question, answer=answer)

def ask_question(question):
    # Get embeddings for the user question
    question_embedding = get_embeddings(question)

    # Perform a similarity search using Pinecone
    query_result = index.query(
        vector=question_embedding.tolist(),  # Ensure the vector is a list of floats
        top_k=1,  # Get the most similar document
        include_metadata=True
    )

    # Fetch the most relevant document
    best_match = query_result['matches'][0]['id']
    response = f"Best Match: {documents[int(best_match.split('_')[1])]}"
    
    # Use OpenAI's chat model (gpt-3.5-turbo) to further generate a response
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the new supported chat model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer the following question based on the document: {documents[int(best_match.split('_')[1])]}\nQuestion: {question}"}
        ],
        max_tokens=150
    )
    
    refined_answer = completion['choices'][0]['message']['content'].strip()
    return refined_answer

if __name__ == "__main__":
    app.run(debug=True)
