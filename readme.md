Add .env file with these variables

```
OPENAI_API_TYPE=azure
OPENAI_API_KEY=
OPENAI_API_BASE=
OPENAI_API_VERSION=2023-03-15-preview

#HuggingFace API
HUGGINGFACE_API_KEY=

#used by the agent for saving images
#for windows
IMAGES_USER_FOLDER=C:\Users\{REPLACE_WITH_YOUR_USERNAME}\Downloads\
```

run fastapi

```
# for streaming chain
uvicorn --app-dir=. api.streaming_chain:app --reload

# for streaming agent
# the agent uses hf endpoint for creating image and saving locally.
uvicorn --app-dir=. api.streaming_agent:app --reload


```

test with python code

```
test_scripts/streaming.py
```

test with html
Open the file in browser (no server needed)

```
index.html
```
