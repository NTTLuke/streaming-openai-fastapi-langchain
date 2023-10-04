Add .env file with these variables

```
OPENAI_API_TYPE=azure
OPENAI_API_KEY=
OPENAI_API_BASE=
OPENAI_API_VERSION=2023-03-15-preview
```

run fastapi

```
uvicorn --app-dir=. api.streaming:app --reload
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
