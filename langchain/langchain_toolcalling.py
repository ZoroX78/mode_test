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
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Local Ollama model — change to any model from `ollama list`
# Examples: "gemma4:e2b", "llama3.2", "qwen2.5", etc.
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

model = ChatOllama(
    model=MODEL_NAME,
    base_url="http://127.0.0.1:11434",
)




# Web page
loader = WebBaseLoader("https://www.geeksforgeeks.org/courses?source=google&medium=cpc&device=c&keyword=gfg&matchtype=b&campaignid=20039445781&adgroup=147845288105&gad_source=1&gad_campaignid=20039445781&gbraid=0AAAAAC9yBkAwHog5Pv03T3awUqCqpJqII&gclid=Cj0KCQjwm8bTBhDWARIsAC9Hi8m216JKp60q4PykKjhOrU6zBTnaQTHlnWq39CYQEYC1mRF8q-YLU4IaAhh9EALw_wcB")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # maximum characters per chunk
    chunk_overlap=50,      # overlapping characters between consecutive chunks
    separators=["\n\n", "\n", " ", ""]
)

splits = text_splitter.split_documents(docs) 