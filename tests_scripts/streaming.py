import json
import requests

url = "http://localhost:8000/stream_chat/"
message = "Write me a song with 5 phrases"
data = {"content": message}

headers = {"Content-Type": "application/json"}

with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r:
    for chunck in r.iter_content(1024):
        print(chunck)
