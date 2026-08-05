import os

# OLLAMA_HOST is sometimes set to a bind address like 0.0.0.0:11434 (server-only).
# Clients must connect to 127.0.0.1, so override a bad host for this process.
_host = os.environ.get("OLLAMA_HOST", "")
if _host.startswith("0.0.0.0"):
    os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Local Ollama model — change to any model from `ollama list`
# Examples: "gemma4:e2b", "llama3.2", "qwen2.5", etc.
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

llm = ChatOllama(
    model=MODEL_NAME,
    base_url="http://127.0.0.1:11434",
)


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True})

db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use only the provided context to answer the user's question. If you do not know the answer, say 'I don't know.'\n\nContext:\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("How do I update my payment method?")
print(response)