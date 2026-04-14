# deeplake_llamaindex

Minimal example of building a DeepLake-backed vector store using LlamaIndex.

## Setup

```powershell
cd deeplake_llamaindex
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

(Optional) copy `.env.example` to `.env` and set:

## Run

```powershell
.\.venv\Scripts\python.exe -u .\scripts\create_deeplake_vectorstore.py
```
