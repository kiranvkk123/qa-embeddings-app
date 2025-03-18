# QA Embeddings App

A Flask-based web application that answers questions by searching a knowledge base using OpenAI embeddings and Pinecone vector search.

## Features

- **Question Answering**: Users can ask questions, and the app provides an answer by matching the query to relevant documents.
- **Pinecone Integration**: Uses Pinecone for fast, similarity-based search of document embeddings.
- **OpenAI Integration**: Leverages OpenAI's GPT-3.5 for refining the responses based on the most relevant document.
- **Document Embedding**: The app generates embeddings of documents using OpenAI's API for similarity comparison.

## Requirements

Before running the app, make sure to install the necessary dependencies:

- Python 3.7+
- Flask
- OpenAI API Key
- Pinecone API Key

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/qa-embeddings-app.git
   cd qa-embeddings-app
Set up a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set up your OpenAI and Pinecone API keys:

Replace 'your-openai-api-key' and 'your-pinecone-api-key' in app.py with your actual API keys.
You can get the OpenAI API key from OpenAI's API page.
You can get the Pinecone API key by signing up at Pinecone.io.
Run the Flask app:

bash
Copy
Edit
python app.py
Open your browser and go to http://127.0.0.1:5000/ to start asking questions!

Usage
Once the app is running, open a web browser and go to http://127.0.0.1:5000/. You'll see a simple form where you can type in your question. The app will process your question, search the knowledge base using Pinecone, and then use OpenAI's GPT-3.5 to provide a refined answer.

How it works:
Document Embedding: The app generates embeddings for each document (like "What is Data Science?") using OpenAI's API.
Question Embedding: When a user asks a question, the app generates an embedding for the question.
Similarity Search: It then uses Pinecone to find the most similar document to the question.
Response Generation: Finally, OpenAI's GPT-3.5 is used to generate an answer based on the most relevant document.
Project Structure
bash
Copy
Edit
qa-embeddings-app/
├── app.py            # Flask app code
├── requirements.txt  # Python dependencies
├── templates/
│   └── index.html    # HTML template for the app
└── .gitignore        # Files to ignore (e.g., virtual environment, sensitive files)
└── README.md         # Project documentation
app.py
Contains the logic for setting up the Flask application, connecting to OpenAI and Pinecone, embedding documents, and generating answers based on user queries.

requirements.txt
Lists the dependencies for the project, which include Flask, OpenAI, Pinecone, and others.

templates/index.html
A simple HTML template for rendering the question and displaying the answer on the web page.
