from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

# Load documents
print("Loading documents...")
loader = TextLoader("data/docs.txt", encoding="utf-8")
documents = loader.load()

# Split into chunks
print("Splitting documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Create embeddings and vector store
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Set up LLM and QA chain
print("Setting up LLM...")
llm = ChatGroq(
    model="llama3-8b-8192",  
    api_key=os.getenv("GROQ_API_KEY")
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
# Interactive Q&A loop
print("\nEDA RAG Assistant ready! Type 'exit' to quit.\n")
while True:
    question = input("Ask a question: ")
    if question.lower() == "exit":
        break
    answer = qa_chain.invoke(question)
    print(f"\nAnswer: {answer['result']}\n")
