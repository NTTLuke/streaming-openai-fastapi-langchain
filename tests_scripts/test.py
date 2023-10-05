import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


image = Image.open("C:\\Users\\nttLu\\Downloads\\a_dog_with_a_hat.png")

local_name = f"image.png"
user_folder = os.getenv("IMAGES_USER_FOLDER")
path = Path(user_folder)

file_path = path / local_name

print("file_path", file_path)

image.save(file_path)
