from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.vector_stores.deeplake import DeepLakeVectorStore


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. LlamaIndex needs an embedding model to write to "
            "the vector store. Set OPENAI_API_KEY in your environment/.env (or configure "
            "a non-OpenAI embedding model)."
        )

    # Where to store your DeepLake vector store.
    # - For Deep Lake Cloud: set DEEPLAKE_DATASET_PATH to e.g. hub://<org>/<dataset>
    # - For local testing: mem://something (in-memory, non-persistent)
    dataset_path = os.getenv("DEEPLAKE_DATASET_PATH", "mem://llamaindex_intro")

    loader = WikipediaReader()
    documents = loader.load_data(
        pages=["Natural Language Processing", "Artificial Intelligence"],
    )
    print("documents:", len(documents))

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)
    print("nodes:", len(nodes))

    vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # This will embed nodes (uses default embedding model) and store them in DeepLake.
    _index = VectorStoreIndex(nodes, storage_context=storage_context)
    del _index

    print("vector_store_nodes:", len(vector_store.get_nodes()))


if __name__ == "__main__":
    main()
