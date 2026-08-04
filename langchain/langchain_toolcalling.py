import os 

os.environ["NVIDIA_API_KEY"] = "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"

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser

# Chain setup with updated model ID
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

model = ChatNVIDIA(model="deepseek-ai/deepseek-v4-pro")
output_parser = StrOutputParser()
chain = prompt | model | output_parser

result = chain.invoke({"topic": "chickens"})
print(result)
