import qdrant_client
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import qdrant
from readpdf import read_pdf_files
from dotenv import load_dotenv

load_dotenv()

def add_data(folder_path):
    try:
        # Extract pdfs and get text chunks 
        text_chunks = read_pdf_files(folder_path)
        print('Text extracted and chunks created successfully.')
        print("The length of the chunks is", len(text_chunks))

        # Create Qdrant client instance with API key retrieved from environment variables
        client = qdrant_client.QdrantClient(
            os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Specify the embedding model for converting text chunks to numerical embedding
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Create a vector store (Qdrant) to store the embeddings of text chunks
        vector_store = qdrant.Qdrant(
            client=client, 
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
            embeddings=embeddings,
        )

        # Add the text chunks extracted from PDF files to the vector store
        vector_store.add_texts(text_chunks)
        print("Data added Successfully")

    except Exception as error:
        print(f"Error: {error}")

# Provide the directory where the pdfs are stored
path = os.getenv("DOCUMENTS_PATH")
add_data(path)