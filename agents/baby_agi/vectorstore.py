from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Initialize the vectorstore as empty
import faiss


# Define your embedding model
EMBEDDINGS_MODEL = OpenAIEmbeddings()

EMBEDDING_SIZE = 1536
index = faiss.IndexFlatL2(EMBEDDING_SIZE)

def get_vectorstore():
    return FAISS(EMBEDDINGS_MODEL.embed_query, index, InMemoryDocstore({}), {})