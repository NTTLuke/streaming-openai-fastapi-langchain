import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.retrievers import ParentDocumentRetriever
from langchain.vectorstores import Chroma
from langchain.vectorstores.pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader
from llms.llm import azure_openai_embeddings

from azure.storage.blob import BlobServiceClient
from io import BytesIO
from pypdf import PdfReader
import logging

from dotenv import load_dotenv
import pinecone

import io
import logging
from pypdf import PdfReader, PdfWriter
from azure.storage.blob import BlobServiceClient

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)


def get_blob_bytes(
    blob_service_client: BlobServiceClient, container_name: str, blob_name: str
) -> bytes:
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall()


def process_blob_page(
    blob_bytes: bytes, page_number: int, blob_name: str, container_name: str
):
    bytes_stream = BytesIO(blob_bytes)
    reader = PdfReader(bytes_stream)
    page_text = reader.pages[0].extract_text()

    metadata = {
        "page_number": page_number,
        "blob_name": blob_name,
        "container_name": container_name,
    }

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    text_chunks = text_splitter.split_text(page_text)
    docs = text_splitter.create_documents(text_chunks, metadata)
    return docs


def save_on_pinecone(
    index_name: str,
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_name: str,
) -> bool:
    """
    Download the pdf from Azure Blob Storage, chunk them, and save on Pinecone.
    Parameters:
        index_name (str): The name of the Pinecone index.
        blob_service_client (BlobServiceClient): The Azure Blob Service client.
        container_name (str): The name of the Azure Blob container.
        blob_name (str): The name of the blob in Azure Blob Storage.
    Returns:
        bool: A boolean indicating whether the operation was successful.
    """

    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_pages = container_client.list_blobs(name_starts_with=blob_name)
        total_docs = []
        page_number = 1

        for blob_page in blob_pages:
            blob_bytes = get_blob_bytes(
                blob_service_client, container_name, blob_page.name
            )
            docs = process_blob_page(
                blob_bytes, page_number, blob_page.name, container_name
            )
            total_docs.extend(docs)
            page_number += 1

        # TODO: Save total_docs on Pinecone
        # ...

        logging.info(
            f"Processed {page_number - 1} pages and prepared documents for Pinecone."
        )
        return True
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False
