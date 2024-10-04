from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3", request_timeout=120)
Settings.llm = llm

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

store_path = "./db_indexing"
index.storage_context.persist(store_path)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)

response = query_engine.query("Name some Electron Beam Techniques")

print(response)
