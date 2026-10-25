import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA

print("Loading document...")
# 2. Load the PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 3. Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Split document into {len(chunks)} chunks.")

# 4. Create embeddings and store in ChromaDB
print("Creating vector database (this may take a moment)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 5. Set up the LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# 6. Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

print("\n--- RAG Chatbot is Ready! ---")
print("Type 'quit' to exit.\n")

# 7. The Chat Loop
# 7. The Chat Loop
while True:
    query = input("Ask a question about the document: ")
    if query.lower() in ['quit', 'exit']:
        break
    
    print("\nThinking... (this may take 10-20 seconds)")
    
    # This streams the answer word-by-word so you can see it working
    for chunk in qa_chain.stream({"query": query}):
        if "result" in chunk:
            print(chunk["result"], end="", flush=True)
    print("\n")
