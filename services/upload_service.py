from io import BytesIO
from pypdf import PdfReader, PdfWriter
from azure.storage.blob import BlobServiceClient
import logging


class UploadService:
    def __init__(self, connection_string):
        self.conn_str = connection_string
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def _check_or_create_container(
        self, blob_service_client: BlobServiceClient, container_name: str
    ) -> None:
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()
            self.logger.info(f"Container {container_name} created.")

    def _upload_page_to_blob(
        self,
        blob_service_client: BlobServiceClient,
        container_name: str,
        file_name: str,
        page_num: int,
        page_data: bytes,
    ) -> None:
        blob_name = f"{file_name}-{page_num + 1}.pdf"
        blob_client = blob_service_client.get_blob_client(container_name, blob_name)
        blob_client.upload_blob(page_data, overwrite=True)
        self.logger.info(
            f"Page {page_num + 1} uploaded to {container_name}/{blob_name}."
        )

    def split_and_upload_pdf(
        self,
        pdf_file: bytes,
        file_name: str,
        container_name: str,
        blob_service_client: BlobServiceClient,
    ) -> int:
        """Split the PDF into pages and upload them to Azure Blob Storage."""
        self._check_or_create_container(blob_service_client, container_name)

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

                self._upload_page_to_blob(
                    blob_service_client, container_name, file_name, page_num, f
                )

        self.logger.info("PDF pages uploaded to Azure Blob Storage.")
        return page_numbers

    def upload(
        self,
        file: bytes,
        file_name: str,
        container_name: str,
        blob_service_client: BlobServiceClient,
    ) -> bool:
        self._check_or_create_container(blob_service_client, container_name)

        blob_name = file_name
        blob_client = blob_service_client.get_blob_client(container_name, blob_name)
        blob_client.upload_blob(file, overwrite=True)
        self.logger.info(f"Document uploaded to {container_name}/{blob_name}.")

        return True
