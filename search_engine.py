from langchain_core.documents import Document
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

from langchain_community.document_loaders import PyPDFLoader
file_path = './nke-10k-2023.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text:v1.5')
sample_embedding = embeddings.embed_query("test")
embedding_size = len(sample_embedding)

from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
client = QdrantClient(
    host='localhost',
    port=6333,
)

is_new = True
try:
    client.get_collection(
        collection_name='test',
    )
    is_new = False
except:
    client.create_collection(
        collection_name='test',
        vectors_config=models.VectorParams(
            size=embedding_size,
            distance=models.Distance.COSINE
        )
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name='test',
    embedding=embeddings,
)

if is_new:
    vector_store.add_documents(documents=all_splits)

# results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")
# print(results[0])
# results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
# doc, score = results[0]
# print(f"Score: {score}\n")
# print(doc)
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print(results[0])
