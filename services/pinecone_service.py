from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os
from pypdf import PdfReader
import logging
from langchain.vectorstores.pinecone import Pinecone
import pinecone
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


class PineconeService:
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.embeddings = embeddings

        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
        )

    def _get_blob_bytes(
        self,
        blob_service_client: BlobServiceClient,
        container_name: str,
        blob_name: str,
    ) -> bytes:
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.download_blob().readall()

    def _chunk_pdf_blob_page(
        self, blob_bytes: bytes, page_number: int, blob_name: str, container_name: str
    ):
        bytes_stream = BytesIO(blob_bytes)
        reader = PdfReader(bytes_stream)
        page_text = reader.pages[0].extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text_chunks = text_splitter.split_text(page_text)
        docs = text_splitter.create_documents(text_chunks)

        for doc in docs:
            doc.metadata.update({"page_number": page_number})
            doc.metadata.update({"blob_name": blob_name})
            doc.metadata.update({"container_name": container_name})

        return docs

    def save_pdf_on_pinecone(
        self,
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
            blob_name (str): The suffix name of blobs in Azure Blob Storage.
        Returns:
            bool: A boolean indicating whether the operation was successful.
        """

        try:
            container_client = blob_service_client.get_container_client(container_name)

            # load only the blobs that start with the blob_name which are the pages of the main pdf
            blob_pages = container_client.list_blobs(name_starts_with=blob_name)
            total_docs = []
            page_number = 1

            for blob_page in blob_pages:
                blob_bytes = self._get_blob_bytes(
                    blob_service_client, container_name, blob_page.name
                )
                docs = self._chunk_pdf_blob_page(
                    blob_bytes, page_number, blob_page.name, container_name
                )
                total_docs.extend(docs)
                page_number += 1

            # save documents on pinecone
            Pinecone.from_documents(
                documents=total_docs,
                embedding=self.embeddings,
                index_name=index_name,
                # namespace="default", #not supported with started plan
            )

            self.logger.info(
                f"Processed {page_number - 1} pages and saved documents to Pinecone."
            )
            return True
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return False

    def save_docx_on_pinecone(
        self,
        index_name: str,
        blob_service_client: BlobServiceClient,
        container_name: str,
        blob_name: str,
    ) -> bool:
        """
        Download the docx from Azure Blob Storage, chunk it, and save on Pinecone.
        Parameters:
            index_name (str): The name of the Pinecone index.
            blob_service_client (BlobServiceClient): The Azure Blob Service client.
            container_name (str): The name of the Azure Blob container.
            blob_name (str): The suffix name of blobs in Azure Blob Storage.
        Returns:
            bool: A boolean indicating whether the operation was successful.
        """

        try:
            from azure.storage.blob import (
                BlobServiceClient,
                BlobClient,
                generate_blob_sas,
                BlobSasPermissions,
            )

            blob_bytes = self._get_blob_bytes(
                blob_service_client, container_name, blob_name
            )

            self._load_documents(blob_bytes)

            # chunk them
            # save on pinecone

            # save documents on pinecone
            # Pinecone.from_documents(
            #     documents=total_docs,
            #     embedding=azure_openai_embeddings(),
            #     index_name=index_name,
            #     # namespace="default",
            # )

            return True
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return False

    def _load_documents(self, blob_bytes):
        # see https://saeedesmaili.com/demystifying-text-data-with-the-unstructured-python-library/
        # see https://github.com/Farhad-Davaripour/DocsGPT/blob/main/docsGpt.py

        # different approaches, choose one:
        # - base text extraction on docx,
        #       cons: without page number, no way to retrieve the page number from UI
        # - convert to pdf and use the same approach as pdf.
        #       cons we have to convert to pdf (we can do at runtime or user can convert?) so we have to work on a copy (need blob storage, we don't work on the original file)
        # - use unstructured library,
        #       cons: lib is very big so the docker image will be big
        # - any suggestions ?

        from docx import Document

        bytes_stream = BytesIO(blob_bytes)

        doc = Document(bytes_stream)

        raw_text = ""
        for paragraph in doc.paragraphs:
            raw_text += paragraph.text

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text_chunks = text_splitter.split_text(raw_text)
        docs = text_splitter.create_documents(text_chunks)

        # IT's not possible retrieve the page number from docx
        # for doc in docs:
        #     doc.metadata.update({"page_number": page_number})
        #     doc.metadata.update({"blob_name": blob_name})
        #     doc.metadata.update({"container_name": container_name})

        return docs


if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llms.llm import azure_openai_embeddings
    from dotenv import load_dotenv

    load_dotenv()

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    pinecone_service = PineconeService(azure_openai_embeddings())
    pinecone_status = pinecone_service.save_docx_on_pinecone(
        blob_service_client=blob_service_client,
        container_name="test-luke",
        index_name="test-luke",
        blob_name="sample.docx",
    )
