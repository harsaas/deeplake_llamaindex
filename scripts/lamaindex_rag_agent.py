import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from llama_index.core import SimpleDirectoryReader
tesla_docs = SimpleDirectoryReader( input_files=["./data/1k/tesla.txt"] ).load_data()

doc0 = tesla_docs[0]
print(doc0.get_content())

from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage
my_activeloop_org_id = "hariprasad86"
my_activeloop_dataset_name = "LlamaIndex_tesla_predictions"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

#hub://hariprasad86/LlamaIndex_tesla_predictions
vectorstore = DeepLakeVectorStore(dataset_path=dataset_path)
storage_context = StorageContext.from_defaults(vector_store=vectorstore)
tesla_index = VectorStoreIndex.from_documents(tesla_docs, storage_context=storage_context)
print("Loaded the  tesla_index index.")
webtext_docs = SimpleDirectoryReader( input_files=["./data/1k/web.txt"] ).load_data()
storage_context_webtext = StorageContext.from_defaults(vector_store=vectorstore)
webtext_index = VectorStoreIndex.from_documents(webtext_docs, storage_context=storage_context_webtext)

print("Loaded the webtext index.")

#Query engine

tesla_engine = tesla_index.as_query_engine(similarity_top_k=3)
webtext_engine = webtext_index.as_query_engine(similarity_top_k=3)
# define the Query engine tool and tool metadata
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
tesla_tool = QueryEngineTool.from_defaults(
    query_engine=tesla_engine,
    name="tesla_index_query_engine",
    description="use this tool to answer questions about tesla predictions",
)
webtext_tool = QueryEngineTool.from_defaults(
    query_engine=webtext_engine,
    name="webtext_index_query_engine",
    description="use this tool to answer questions about webtext data",
)

query_engine_tools = [tesla_tool, webtext_tool]

#Agent from llama_index.agents import AgentExecutor, ZeroShotAgent
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
agent = OpenAIAgent.from_tools(
    tools=query_engine_tools,
    llm=llm,
    system_prompt="You are a helpful assistant that answers questions about tesla predictions and webtext data.",
    response_format="text",)

#Now that we have our agent, we can execute an interactive chat interface (REPL, Read-Eval-Print Loop) where the agent can receive inputs (like questions or prompts), process them, and return responses, making it a conversational agent capable of handling a dialogue or chat session

response = agent.chat("What are the tesla predictions for 2025? and how its compared to 2024?")
print(response)