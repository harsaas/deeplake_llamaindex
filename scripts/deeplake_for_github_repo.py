from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.vector_stores.deeplake import DeepLakeVectorStore


def _parse_github_url(url: str) -> tuple[str | None, str | None]:
    match = re.match(r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _get_default_branch(owner: str, repo: str, github_token: str | None) -> str | None:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    if github_token:
        req.add_header("Authorization", f"Bearer {github_token}")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ):
        return None

    default_branch = payload.get("default_branch")
    return default_branch if isinstance(default_branch, str) and default_branch else None


def main() -> None:
    nest_asyncio.apply()
    load_dotenv()

    github_repo_url = os.getenv(
        "GITHUB_REPO_URL",
        "https://github.com/harsaas/mylanggraphlearning",
    )
    owner, repo = _parse_github_url(github_repo_url)
    if not owner or not repo:
        raise ValueError(
            f"Invalid GITHUB_REPO_URL: {github_repo_url}. Expected https://github.com/<owner>/<repo>"
        )

    github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GIT_HUB_TOKEN")
    if not github_token:
        raise RuntimeError(
            "Missing GitHub token. Set GITHUB_TOKEN=... (preferred) or GIT_HUB_TOKEN=... in your .env"
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set OPENAI_API_KEY=... in your .env so embeddings/LLM calls can be made."
        )

    dataset_path = os.getenv(
        "DEEPLAKE_DATASET_PATH",
        "hub://hariprasad86/repository_vector_store",
    )

    branch = os.getenv("GITHUB_BRANCH")
    if not branch:
        branch = _get_default_branch(owner, repo, github_token) or "main"

    github_client = GithubClient(github_token=github_token, verbose=False)

    reader = GithubRepositoryReader(
        github_client=github_client,
        owner=owner,
        repo=repo,
        filter_file_extensions=(
            [".py", ".js", ".ts", ".md"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        verbose=False,
        concurrent_requests=5,
    )

    print(f"Loading repo {owner}/{repo} (branch={branch})")
    documents = reader.load_data(branch=branch)
    print(f"documents: {len(documents)}")

    vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Uploading to DeepLake: {dataset_path}")
    vector_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    print(f"vector_store_nodes: {len(vector_store.get_nodes())}")

    query_engine = vector_index.as_query_engine()
    intro_question = "What is the repository about?"
    print(f"Test question: {intro_question}")
    print("=" * 50)
    answer = query_engine.query(intro_question)
    print(f"Answer: {textwrap.fill(str(answer), 100)}\n")

    if not sys.stdin.isatty():
        return

    while True:
        user_question = input("Please enter your question (or type 'exit' to quit): ")
        if user_question.strip().lower() in {"exit", "quit"}:
            print("Exiting, thanks for chatting!")
            break
        if not user_question.strip():
            continue

        print(f"Your question: {user_question}")
        print("=" * 50)
        answer = query_engine.query(user_question)
        print(f"Answer: {textwrap.fill(str(answer), 100)}\n")


if __name__ == "__main__":
    main()
