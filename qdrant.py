from langchain_community.vectorstores import Qdrant
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import os
from readpdf import read_pdf_files

load_dotenv()

try:
    # Create Qdrant client instance with API key retrieved from ENV
    client = QdrantClient(
            os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY")
        )


    # Customize vector configuration
    collection_config = VectorParams(
            size=1536, # 1536 for OpenAI
            distance=Distance.COSINE
        )
    
    # Create collection with API key retrieved from ENV
    client.create_collection(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        vectors_config=collection_config
    )

    print('Client created Successfully')

except Exception as error:
    print(f"Error: {error}")
