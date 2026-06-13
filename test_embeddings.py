from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
db = InMemoryVectorStore(embeddings)

db.add_texts(["Hello world", "Machine learning is cool"])
results = db.similarity_search("AI", k=1)
print(results)
