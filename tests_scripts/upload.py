import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.retrievers import ParentDocumentRetriever
from langchain.vectorstores import Chroma
from langchain.vectorstores.pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader
from llms.llm import azure_openai_embeddings

from dotenv import load_dotenv
import pinecone
from io import BytesIO
import logging
from pypdf import PdfReader, PdfWriter
from azure.storage.blob import BlobServiceClient

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO .. replace InMemory with blob storage
def save_on_pinecone_as_parent_child(blob_service_client, container_name, blob_name):
    """Parent Document Retriever pattern implementation"""
    loaders = [
        TextLoader("./resources/state_of_the_union.txt", encoding="utf-8"),
    ]

    # TODO:
    # Upload to azure blob storage
    pdf_path = "./resources/US-constitution.pdf"
    file_name = "US-constitution"

    # Check if the container exists, and create it if it doesn't
    if not blob_service_client.get_container_client(container_name).exists():
        blob_service_client.create_container(container_name)

    container_client = blob_service_client.get_container_client(
        container=container_name
    )

    with open(file=filepath, mode="rb") as data:
        blob_client = container_client.upload_blob(
            name=blob_name, data=data, overwrite=True
        )

    # get all documents from azure blob storage given a container name

    docs = []
    for l in loaders:
        docs.extend(l.load())

    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # The vectorstore to use to index the child chunks
    vectorstore = Pinecone.from_existing_index(
        index_name="parent-demo-idx",
        embedding=azure_openai_embeddings(),
    )

    # The storage layer for the parent documents
    # TODO: replace with Redis
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(docs)

    print(len(list(store.yield_keys())))

    # Let's make sure the underlying vector store still retrieves the small chunks.
    sub_docs = vectorstore.similarity_search("justice breyer")
    print(sub_docs[0].page_content)

    retrieved_docs = retriever.get_relevant_documents("justice breyer")
    print(len(retrieved_docs[0].page_content))
    print(retrieved_docs[0].page_content)


def _check_or_create_container(
    blob_service_client: BlobServiceClient, container_name: str
) -> None:
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()
        logger.info(f"Container {container_name} created.")


def _upload_page_to_blob(
    blob_service_client: BlobServiceClient,
    container_name: str,
    file_name: str,
    page_num: int,
    page_data: bytes,
) -> None:
    blob_name = f"{file_name}-{page_num + 1}.pdf"
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    blob_client.upload_blob(page_data, overwrite=True)
    logger.info(f"Page {page_num + 1} uploaded to {container_name}/{blob_name}.")


def split_and_upload_pdf(
    pdf_file: bytes,
    file_name: str,
    container_name: str,
    blob_service_client: BlobServiceClient,
) -> int:
    _check_or_create_container(blob_service_client, container_name)

    with BytesIO(pdf_file) as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        page_numbers = len(pdf_reader.pages)

        for page_num in range(page_numbers):
            pdf_page = pdf_reader.pages[page_num]

            f = BytesIO()
            writer = PdfWriter()
            writer.add_page(pdf_page)
            writer.write(f)
            f.seek(0)

            _upload_page_to_blob(
                blob_service_client, container_name, file_name, page_num, f
            )

    logger.info("PDF pages uploaded to Azure Blob Storage.")
    return page_numbers


def upload_to_storage(
    word_file: bytes,
    file_name: str,
    container_name: str,
    blob_service_client: BlobServiceClient,
) -> bool:
    _check_or_create_container(blob_service_client, container_name)

    blob_name = f"{file_name}.docx"
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    blob_client.upload_blob(word_file, overwrite=True)
    logger.info(f"Document uploaded to {container_name}/{blob_name}.")

    return True


# def save_on_pinecone(
#     index_name: str,
#     blob_service_client: BlobServiceClient,
#     container_name: str,
#     blob_name: str,
# ) -> bool:
#     from io import BytesIO

#     """Download the pdf from azure blob storage , chunk them and save on pinecone"""

#     # Get the container client
#     container_client = blob_service_client.get_container_client(container_name)

#     # get only the blobs that start with the blob_name
#     blob_pages = container_client.list_blobs(name_starts_with=blob_name)
#     page_number = 1
#     total_docs = []

#     # iterate over the blobs for creating chunks and embedding them
#     for blob_page in blob_pages:
#         blob_client = container_client.get_blob_client(blob_page.name)
#         blob_bytes = blob_client.download_blob().readall()

#         # convert bytes to stream
#         bytes_stream = BytesIO(blob_bytes)

#         # load pdf page from stream
#         reader = PdfReader(bytes_stream)
#         page_text = reader.pages[0].extract_text()

#         metadata = {
#             "page_number": page_number,
#             "blob_name": blob_page.name,
#             "container_name": container_name,
#         }

#         # load documents from text
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=100,
#             chunk_overlap=20,
#         )

#         # split text into chunks
#         text_chunks = text_splitter.split_text(page_text)
#         # create documents from chunks
#         docs = text_splitter.create_documents(text_chunks, metadata)

#         total_docs.extend(docs)

#         page_number += 1

#     # save documents on pinecone
#     Pinecone.from_documents(
#         documents=total_docs,
#         embedding=azure_openai_embeddings(),
#         index_name=index_name,
#         # namespace="default",
#     )
#     return True


def _get_blob_bytes(
    blob_service_client: BlobServiceClient, container_name: str, blob_name: str
) -> bytes:
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall()


def _process_blob_page(
    blob_bytes: bytes, page_number: int, blob_name: str, container_name: str
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


def _save_docx_on_pinecone(
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
        from datetime import datetime, timedelta
        from langchain.document_loaders import Docx2txtLoader
        from azure.storage.blob import (
            BlobServiceClient,
            BlobClient,
            generate_blob_sas,
            BlobSasPermissions,
        )

        # TODO get sas url from blob storage
        start_time = datetime.utcnow() - timedelta(minutes=10)
        expiry_time = start_time + timedelta(days=1)
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry_time,
            start=start_time,
        )

        loader = Docx2txtLoader()
        # chunk them
        # save on pinecone

        # save documents on pinecone
        # Pinecone.from_documents(
        #     documents=total_docs,
        #     embedding=azure_openai_embeddings(),
        #     index_name=index_name,
        #     # namespace="default",
        # )

        logging.info(
            f"Processed {page_number - 1} pages and saved documents to Pinecone."
        )
        return True
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False


def _save_on_pinecone(
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

        # load only the blobs that start with the blob_name
        blob_pages = container_client.list_blobs(name_starts_with=blob_name)
        total_docs = []
        page_number = 1

        for blob_page in blob_pages:
            blob_bytes = _get_blob_bytes(
                blob_service_client, container_name, blob_page.name
            )
            docs = _process_blob_page(
                blob_bytes, page_number, blob_page.name, container_name
            )
            total_docs.extend(docs)
            page_number += 1

        # save documents on pinecone
        Pinecone.from_documents(
            documents=total_docs,
            embedding=azure_openai_embeddings(),
            index_name=index_name,
            # namespace="default",
        )

        logging.info(
            f"Processed {page_number - 1} pages and saved documents to Pinecone."
        )
        return True
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False


def start_load_pdf():
    # 1. split pdf into pages and upload to azure blob storage
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Read the local PDF file into a bytes object
    file_path = "./resources/sample.pdf"
    pdf_file_bytes = None
    with open(file_path, "rb") as file:
        pdf_file_bytes = file.read()

    container_name = "test-luke"
    file_name = "sample"
    page_count = split_and_upload_pdf(
        pdf_file_bytes, file_name, container_name, blob_service_client
    )

    # Print the number of pages uploaded
    print(f"{page_count} pages uploaded to Azure Blob Storage.")

    # 2. Load the documents from azure blob storage in order to chunk and embed them
    index_name = "test-luke"
    _save_on_pinecone(
        index_name=index_name,
        blob_service_client=blob_service_client,
        container_name=container_name,
        blob_name=file_name,
    )

    print("Documents saved on Pinecone.")


def start_load_word():
    from langchain.document_loaders import Docx2txtLoader

    # 1. split pdf into pages and upload to azure blob storage
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Read the local PDF file into a bytes object
    file_path = "./resources/Sample.docx"
    doc_file_bytes = None
    with open(file_path, "rb") as file:
        doc_file_bytes = file.read()

    # loader = Docx2txtLoader("example_data/fake.docx")

    container_name = "test-luke"
    file_name = "sample"
    is_ok = upload_to_storage(
        doc_file_bytes, file_name, container_name, blob_service_client
    )

    print(is_ok)

    index_name = "test-luke"
    _save_docx_on_pinecone(
        index_name=index_name,
        blob_service_client=blob_service_client,
        container_name=container_name,
        blob_name=file_name,
    )

    # # Print the number of pages uploaded
    # print(f"{page_count} pages uploaded to Azure Blob Storage.")

    # # 2. Load the documents from azure blob storage in order to chunk and embed them
    # index_name = "test-luke"
    # _save_on_pinecone(
    #     index_name=index_name,
    #     blob_service_client=blob_service_client,
    #     container_name=container_name,
    #     blob_name=file_name,
    # )

    # print("Documents saved on Pinecone.")


start_load_word()
