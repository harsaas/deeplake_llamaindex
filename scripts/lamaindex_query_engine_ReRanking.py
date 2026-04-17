from __future__ import annotations

import os
import re
import json
import sys
import textwrap
import urllib.error
import urllib.request

import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

from llama_index.core import SimpleDirectoryReader, Settings

# Needed for SubQuestionQueryEngine(use_async=True) in environments where an event loop
# may already be running (e.g., notebooks, some IDE runners).
nest_asyncio.apply()

load_dotenv()

# load documents (the wget step downloads into data/paul_graham/)
documents = SimpleDirectoryReader("data/paul_graham").load_data()

Settings.chunk_size = 512
Settings.chunk_overlap = 64

# Create an index over the documents (embeddings are generated during ingestion)
# NOTE: If the dataset already exists but wasn't created as a DeepLake "VectorStore",
# DeepLake may raise an error unless you use a new path or set overwrite.
dataset_path = os.getenv(
    "DEEPLAKE_DATASET_PATH",
    "hub://hariprasad86/paul_graham_essays",
)
overwrite = os.getenv("DEEPLAKE_OVERWRITE", "false").strip().lower() in {"1", "true", "yes"}

vector_store = DeepLakeVectorStore(
    dataset_path=dataset_path,
    overwrite=overwrite,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

print(f"vector_store_nodes: {len(vector_store.get_nodes())}")

query_engine = vector_index.as_query_engine(streaming=True, similarity_top_k=10)
intro_question = "What does Paul Graham do?"
print(f"Test question: {intro_question}")
print("=" * 50)
response = query_engine.query(intro_question)
response.print_response_stream()
print("\n")

# Sub Question Query Engine for summarization
sub_query_engine = vector_index.as_query_engine(
    streaming=True,
    similarity_top_k=5,
    response_mode="tree_summarize",
)

response_summary = sub_query_engine.query("What are the main points of the essay?")
response_summary.print_response_stream()
print("\n")


from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="pg_essay",
            description="Paul Graham essay on What I Worked On",
        ),
    ),
]

# Create the SubQuestionQueryEngine (no settings argument!)
sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=False,
)

sub_response = sub_question_query_engine.query(
    "How was Paul Grahams life different before, during, and after YC?"
)


print( ">>> The final response:\n", sub_response )

#VectorIndexRetriever fetches the top-k nodes that are most similar to the query. It focuses on relevance and similarity, ensuring the results closely align with the query's intent. It is the approach we used in previous subsections
# Use Case: It is ideal for situations where precision and relevance to the specific query are paramount, like in detailed research or topic-specific inquiries.
# SummaryIndexRetriever retrieves all nodes related to the query without prioritizing their relevance. This approach is less concerned with aligning closely to the specific context of the question and more about providing a broad overview
#Use Case: Useful in scenarios where a comprehensive sweep of information is needed, regardless of the direct relevance to the specific terms of the query, like in exploratory searches or general overviews.

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    print("COHERE_API_KEY not set; skipping Cohere reranking demo.")
else:
    import cohere

    query_cohere = "What is the capital of the United States?"
    documents_cohere = [
        "Carson City is the capital city of the American state of Nevada. At the  2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
        "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. ",
        "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
        "North Dakota is a state in the United States. 672,591 people lived in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
    ]

    # Direct Cohere SDK rerank call (independent of LlamaIndex)
    cohere_client = cohere.Client(cohere_api_key)
    response_cohere = cohere_client.rerank(
        query=query_cohere,
        documents=documents_cohere,
        top_n=3,
        model="rerank-v3.5",
    )
    print(response_cohere)

    results = response_cohere.results
    for rank, r in enumerate(results):
        doc_text = documents_cohere[r.index]
        score = r.relevance_score
        print(f"Document Rank: {rank}")
        print(f"Document: {doc_text}")
        print(f"Relevance Score: {score:.2f}\n")

    # LlamaIndex node postprocessor reranker
    from llama_index.postprocessor.cohere_rerank import CohereRerank

    cohere_reranker = CohereRerank(
        api_key=cohere_api_key,
        model="rerank-v3.5",
        top_n=3,
    )

    query_engine_with_reranking = vector_index.as_query_engine(
        node_postprocessors=[cohere_reranker]
    )
    response_with_reranking = query_engine_with_reranking.query(
        "What is the capital of the United States?"
    )
    print("Response with Cohere Reranking:", response_with_reranking)

    query_engine_cohere_rerank = vector_index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[cohere_reranker],
    )

    response_after_reranking = query_engine_cohere_rerank.query(
        "What did Sam Altman do in this essay?",
    )
    print(response_after_reranking)

