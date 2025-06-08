import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Example LangChain imports
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
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

VECTORSTORE = None
CONVERSATION_CHAIN = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdfs():
    """
    Accept multiple PDF files. Save them to disk and return a list of filenames.
    """
    # Check if 'files' is in the request
    if 'files' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    # We can get multiple files
    files = request.files.getlist('files')
    api_key = request.form.get('api_key', None)

    if not api_key:
        return jsonify({"error": "OpenAI API key is required."}), 400

    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    saved_filenames = []
    for file in files:
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{original_filename}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(save_path)
        saved_filenames.append(unique_filename)

    return jsonify({
        "message": f"Uploaded {len(saved_filenames)} PDF(s) successfully.",
        "filenames": saved_filenames
    }), 200

@app.route('/train', methods=['POST'])
def train_model():
    """
    Build or rebuild the vector store using all uploaded PDFs.
    """
    global VECTORSTORE
    global CONVERSATION_CHAIN

    data = request.get_json()
    filenames = data.get('filenames', [])
    api_key = data.get('api_key', None)

    if not filenames:
        return jsonify({"error": "No filenames provided"}), 400
    if not api_key:
        return jsonify({"error": "OpenAI API key is required"}), 400

    # Set the OpenAI key
    os.environ["OPENAI_API_KEY"] = api_key

    # Aggregate all documents
    docs_all = []
    for fname in filenames:
        pdf_path = os.path.join(UPLOAD_FOLDER, fname)
        if not os.path.exists(pdf_path):
            return jsonify({"error": f"File {fname} not found on server"}), 404

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()
        docs_all.extend(pdf_docs)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs_all)

    # Build the vector store
    embeddings = OpenAIEmbeddings()
    VECTORSTORE = FAISS.from_documents(docs, embeddings)

    # Create chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    retriever = VECTORSTORE.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    CONVERSATION_CHAIN = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return jsonify({"message": "Training completed successfully."}), 200

@app.route('/query', methods=['POST'])
def query_model():
    """
    Use the trained ConversationalRetrievalChain to answer a user query.
    """
    global CONVERSATION_CHAIN

    data = request.get_json()
    user_query = data.get('query', None)
    api_key = data.get('api_key', None)

    if not user_query:
        return jsonify({"error": "Query is required"}), 400
    if not api_key:
        return jsonify({"error": "OpenAI API key is required"}), 400

    # Set key again for each request
    os.environ["OPENAI_API_KEY"] = api_key

    if not CONVERSATION_CHAIN:
        return jsonify({"error": "No model trained. Please upload and train first."}), 400

    # Ask the chain
    result = CONVERSATION_CHAIN({"question": user_query})
    answer = result.get('answer', "No answer found.")

    return jsonify({"answer": answer}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
