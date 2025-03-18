# README: Question-Answering System using Pinecone and OpenAI

## Overview
This project demonstrates a simple question-answering system built with Pinecone, OpenAI, and Flask. It utilizes OpenAI's embeddings to generate vector representations of both documents and user queries, and uses Pinecone for storing and retrieving these embeddings. Flask is used to provide a web interface for users to interact with the system.

## Features
- **Text Embeddings**: Generates embeddings using OpenAI's model.
- **Vector Search**: Uses Pinecone to store and search document embeddings.
- **Question Answering**: Users can input questions, and the system retrieves the most relevant document to answer the question using a fine-tuned response from OpenAI.
- **Web Interface**: A simple Flask web interface allows users to input questions and view answers.

## Prerequisites
Before running the project, make sure you have the following:
- **Python 3.x** installed on your machine.
- **Pinecone account** to access Pinecone's vector database.
- **OpenAI account** to access the API for embeddings and models.
- **Flask** for the web interface.

### Install dependencies
To set up the environment, use `pip` to install the required packages:

pip install openai pinecone-client flask numpy scikit-learn


## Setup

### 1. Pinecone Configuration
Make sure you have a **Pinecone API key** and **OpenAI API key**. Replace `'pinecone key'` and `'your open ai key'` in the script below with your actual API keys.

```python
openai.api_key = 'your open ai key'
pc = Pinecone(api_key="pinecone key", environment="us-east-1")
```

### 2. Pinecone Index Creation
The script checks if the specified index (named `qa-index`) exists. If it does not exist, it creates one. The index is configured to store vector embeddings of dimension 1536, which is the size of OpenAI’s embeddings.

```python
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
```

### 3. Embedding Generation with OpenAI
The `get_embeddings` function is responsible for obtaining embeddings from OpenAI for each document in the knowledge base (and user queries). The embeddings are used to perform similarity searches on Pinecone.

```python
def get_embeddings(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # OpenAI embedding model
        input=text
    )
    embeddings = np.array(response['data'][0]['embedding'])
    return embeddings
```

### 4. Insert Documents into Pinecone
The `insert_embeddings` function preprocesses a list of documents (e.g., data science-related content) by generating their embeddings and inserting them into the Pinecone index.

```python
def insert_embeddings():
    vectors = []
    for i, doc in enumerate(documents):
        embedding = get_embeddings(doc)
        vectors.append((f"doc_{i}", embedding.tolist()))  # Converting numpy array to list
    index.upsert(vectors=vectors)
```

### 5. Flask Web Interface
The Flask app serves a simple web interface. It listens for a question input from the user and uses Pinecone to find the most relevant document. The answer is then generated using OpenAI’s GPT-3.5-turbo model.

```python
@app.route("/", methods=["GET", "POST"])
def home():
    question = request.form.get("question")
    answer = ""
    if question:
        answer = ask_question(question)
    return render_template("index.html", question=question, answer=answer)
```

### 6. Handling User Questions
The `ask_question` function takes the user's question, generates its embedding, and performs a similarity search on the Pinecone index to find the most relevant document. It then uses OpenAI’s chat model to generate a detailed response based on that document.

```python
def ask_question(question):
    question_embedding = get_embeddings(question)
    query_result = index.query(
        vector=question_embedding.tolist(),  # Ensure the vector is a list of floats
        top_k=1,  # Get the most similar document
        include_metadata=True
    )
    best_match = query_result['matches'][0]['id']
    response = f"Best Match: {documents[int(best_match.split('_')[1])]}"
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer the following question based on the document: {documents[int(best_match.split('_')[1])]}\nQuestion: {question}"}
        ],
        max_tokens=150
    )
    
    refined_answer = completion['choices'][0]['message']['content'].strip()
    return refined_answer
```

### 7. Running the Application
To run the application, simply execute the following command:
```bash
python app.py
```

Once the Flask app is running, open your browser and navigate to `http://127.0.0.1:5000/`. You will see a simple interface where you can ask questions, and the system will return the most relevant answer from the knowledge base.

## Folder Structure
The basic structure of the project is as follows:

```
/project-folder
  ├── app.py            # Main application file
  ├── templates/
  │    └── index.html   # HTML template for the web interface
  └── requirements.txt  # List of dependencies
```

## Usage
1. Run the Flask app: `python app.py`
2. Open your browser and navigate to `http://127.0.0.1:5000/`.
3. Type a question related to your knowledge base in the input box.
4. The app will search for the most relevant document in the Pinecone index and generate a response based on that document using OpenAI's model.

## Conclusion
This project showcases how to integrate Pinecone (a vector database) and OpenAI to build an intelligent question-answering system. It demonstrates how to store text embeddings in Pinecone, perform a similarity search, and use OpenAI to refine the responses. You can expand the knowledge base and improve the system by adding more documents and fine-tuning the responses.

---

Let me know if you'd like to add more details or further customize this README!
