import os
import glob
import uuid

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ---------------------------
# Install and import RAG-related libraries
# ---------------------------
# You would typically pip install these:
# !pip install langchain openai tiktoken faiss-cpu pypdf unstructured
# (plus any others you need from the original script)

#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
#from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

app = Flask(__name__)

# Allow PDFs to be uploaded to this directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global references to vector store or chain
VECTORSTORE = None
CONVERSATION_CHAIN = None

@app.route('/')
def index():
    """
    Serve the main HTML page.
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    1. Receives a PDF file.
    2. Saves it in the UPLOAD_FOLDER.
    3. Returns a JSON response with status.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to the server
    filename = secure_filename(file.filename)
    unique_filename = str(uuid.uuid4()) + "_" + filename
    save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(save_path)
    
    return jsonify({
        "message": f"File {filename} uploaded successfully.",
        "filename": unique_filename
    }), 200


@app.route('/train', methods=['POST'])
def train_model():
    """
    1. Takes the filename of the uploaded PDF.
    2. Reads the PDF, splits into chunks, creates a vector store.
    3. Builds a RAG chain with a ConversationBufferMemory for follow-up queries.
    """
    global VECTORSTORE
    global CONVERSATION_CHAIN

    data = request.get_json()
    filename = data.get('filename', None)
    api_key = data.get('api_key', None)

    if not filename:
        return jsonify({"error": "Filename is required."}), 400
    if not api_key:
        return jsonify({"error": "OpenAI API key is required."}), 400

    # Set the user-provided OpenAI key
    os.environ["OPENAI_API_KEY"] = api_key

    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(pdf_path):
        return jsonify({"error": "File not found on server."}), 404

    # 1) Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2) Chunk the PDF text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # 3) Create embeddings and build vector store
    embeddings = OpenAIEmbeddings()
    VECTORSTORE = FAISS.from_documents(docs, embeddings)

    # 4) Create a ConversationalRetrievalChain for question-answering
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    retriever = VECTORSTORE.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    CONVERSATION_CHAIN = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return jsonify({"message": "Model trained successfully."}), 200


@app.route('/query', methods=['POST'])
def query_model():
    """
    1. Accepts a query from the user.
    2. Uses the trained ConversationalRetrievalChain to get an answer.
    3. Returns the answer as JSON.
    """
    global CONVERSATION_CHAIN

    data = request.get_json()
    user_query = data.get('query', None)
    api_key = data.get('api_key', None)

    if not user_query:
        return jsonify({"error": "A query is required."}), 400
    if not api_key:
        return jsonify({"error": "OpenAI API key is required."}), 400

    # Set the user-provided OpenAI key on each request
    os.environ["OPENAI_API_KEY"] = api_key

    if not CONVERSATION_CHAIN:
        return jsonify({"error": "No trained model found. Please train first."}), 400

    # Generate the answer with the chain
    result = CONVERSATION_CHAIN({"question": user_query})

    # result['answer'] typically contains the response
    answer = result.get('answer', "No answer found.")

    return jsonify({"answer": answer}), 200


if __name__ == '__main__':
    # You can adjust host/port as needed
    app.run(host='0.0.0.0', port=5000, debug=True)
