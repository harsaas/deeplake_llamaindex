import os
import sys

from dotenv import load_dotenv


load_dotenv()

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex

# build service context
llm = OpenAI(model="gpt-4", temperature=0.0)
service_context = ServiceContext.from_defaults(llm=llm)


vector_store = DeepLakeVectorStore(dataset_path="hub://hariprasad86/paul_graham_essays_llamaindex", overwrite=False, read_only=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# load the index from the vector store
index = VectorStoreIndex.from_vector_store(
	vector_store=vector_store,
	service_context=service_context,
	storage_context=storage_context,
)
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What does Paul Graham do?")
print(response)
#Faithfulness evaluation
from llama_index.core.evaluation import FaithfulnessEvaluator
evaluator = FaithfulnessEvaluator()
eval_result = evaluator.evaluate_response(response=response)
print(eval_result)
print( "> faithfulness evaluator result:", eval_result.passing )

#custom rag evaluation
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_files=["./data/venus_transmission.txt"])

docs = reader.load_data()
print(f"Loaded {len(docs)} docs")

#convert the doc to a node and add to the vector store
from llama_index.core.node_parser import SimpleNodeParser
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(docs)
print(f"Parsed {len(nodes)} nodes")
eval_vector_index = VectorStoreIndex(nodes)
query_engine = eval_vector_index.as_query_engine()

response_vector = query_engine.query("What was The first beings to inhabit the planet?")
print( response_vector.response )

# generate_question_context_pairs class leverages the LLM to create questions based on the content of each node
from llama_index.core.evaluation import generate_question_context_pairs
question_context_pairs = generate_question_context_pairs(nodes=nodes, llm=llm, num_questions_per_chunk=2)
print(f"Generated {len(question_context_pairs.queries)} question-context pairs")
queries = list(question_context_pairs.queries.values())
print( queries[0:10] )

#RetrieverEvaluator class can now use this QA dataset to evaluate the retriever's performance. It queries each question using the retriever and evaluates which chunks are returned as the answer
from llama_index.core.evaluation import RetrieverEvaluator
retriever = eval_vector_index.as_retriever(similarity_top_k=2)

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)
retriever_eval_results = []
for query_id, query in question_context_pairs.queries.items():
    expected_ids = question_context_pairs.relevant_docs.get(query_id, [])
    expected_texts = [
        question_context_pairs.corpus[doc_id]
        for doc_id in expected_ids
        if doc_id in question_context_pairs.corpus
    ]
    retriever_eval_results.append(
        retriever_evaluator.evaluate(
            query=query,
            expected_ids=expected_ids,
            expected_texts=expected_texts,
        )
    )
#print( retriever_eval_results )
import pandas as pd
# function to display eval results in a more readable format
def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
    )

    return metric_df

results_df = display_results("OpenAI Embedding Retriever", retriever_eval_results)
print(results_df.to_string(index=False))