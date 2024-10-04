from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

class EELSEngine:
    def __init__(
        self,
        model_name="llama3",
        embedding_model_name="BAAI/bge-base-en-v1.5",
        index_dir="./db_indexing",
        request_timeout=120,
        streaming=False,
        similarity_top_k=4,
    ):
        """
        Initialize the EELS Engine with the specified LLM, embeddings, and index.

        :param model_name: Name of the model to use with Ollama.
        :param embedding_model_name: Name of the embedding model.
        :param index_dir: Directory where the index is stored.
        :param request_timeout: Timeout for LLM requests in seconds.
        :param streaming: Enable streaming in query responses.
        :param similarity_top_k: Number of top similar documents to retrieve.
        """
        # Initialize the LLM with Ollama
        llm = Ollama(model=model_name, request_timeout=request_timeout)
        Settings.llm = llm

        # Initialize the embedding model
        embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        Settings.embed_model = embed_model

        # Load the index from storage
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        self.index = load_index_from_storage(storage_context)

        # Create the query engine
        self.query_engine = self.index.as_query_engine(
            streaming=streaming, similarity_top_k=similarity_top_k
        )

    def query(self, user_input):
        """
        Process a user query and return the response.

        :param user_input: The user's query as a string.
        :return: The response from the LLM.
        """
        response = self.query_engine.query(user_input)
        return response
