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


