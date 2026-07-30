import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection")

collection.add(
    ids=["1", "2", "3"],
    documents=["This is the first document.", "This is the second document.", "This is the third document."],
)

results = collection.query(
    query_texts=["first document", "second document"],
    n_results=2,)
print(results)