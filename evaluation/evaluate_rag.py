"""Evaluate the hybrid retrieval pipeline defined in 02.retrieval.py.

The pipeline lives in a module whose filename starts with a digit, so it cannot
be imported with a normal ``import`` statement. It is loaded from its file path.
"""


import sys
import types

# ragas 0.4.3 imports ChatVertexAI/VertexAI from langchain-community, but those
# were removed in langchain-community 0.4.x. ragas only uses them for isinstance
# checks, so register placeholders before importing ragas.
try:  # pragma: no cover - depends on installed langchain-community version
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
except ModuleNotFoundError:
    _vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # type: ignore[no-redef]
        pass

    _vertexai_module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_module

    import langchain_community.llms as _langchain_llms

    if not hasattr(_langchain_llms, "VertexAI"):

        class VertexAI:  # type: ignore[no-redef]
            pass

        _langchain_llms.VertexAI = VertexAI


from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

import os

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq


class LocalEmbeddings(Embeddings):
    """FastEmbed embeddings exposing a string ``model`` name.

    ragas records ``embeddings.model`` for usage tracking and requires a string.
    langchain-community's FastEmbedEmbeddings stores the underlying fastembed
    object there, which fails pydantic validation, so this delegates to it while
    exposing a string model name.
    """

    model = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        self._embeddings = FastEmbedEmbeddings()

    def embed_documents(self, texts):
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text):
        return self._embeddings.embed_query(text)


import importlib.util
from pathlib import Path

_RETRIEVAL_PATH = Path(__file__).resolve().parent.parent / "02.retrieval.py"

_spec = importlib.util.spec_from_file_location("retrieval", _RETRIEVAL_PATH)
retrieval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieval)

retrieve_relevant_chunks = retrieval.retrieve_relevant_chunks
generate_answer = retrieval.generate_answer


test_questions = [
    {
        "question": "What is the attention mechanism?",
        "ground_truth": (
            "Attention is a mechanism that allows a model "
            "to focus on different parts of the input."
        )
    },
    {
        "question": "What are queries, keys, and values?",
        "ground_truth": (
            "Queries, keys, and values are representations "
            "used to calculate attention and produce a "
            "weighted combination of values."
        )
    },
    {
        "question": "Why is attention useful?",
        "ground_truth": (
            "Attention allows the model to determine which "
            "parts of the input are important when producing "
            "an output."
        )
    }
]



def run_rag(question):

    # -------------------------
    # Retrieve
    # -------------------------

    results = retrieve_relevant_chunks(
        question,
        top_k=3
    )

    # -------------------------
    # Get contexts
    # -------------------------

    contexts = [
        result["chunk"]
        for result in results
    ]

    # -------------------------
    # Generate answer
    # -------------------------

    answer = generate_answer(
        question,
        results
    )

    return {
        "question": question,
        "contexts": contexts,
        "answer": answer
    }


def create_dataset():

    data = []

    for item in test_questions:

        question = item["question"]

        ground_truth = item["ground_truth"]

        result = run_rag(
            question
        )

        data.append({

            "user_input": question,

            "retrieved_contexts": result["contexts"],

            "response": result["answer"],

            "reference": ground_truth
        })

    return data


def evaluate_rag():

    load_dotenv()

    run_config = RunConfig(
        timeout=300,
        max_workers=1,
        max_retries=2
    )

    llm = LangchainLLMWrapper(
        ChatGroq(
            api_key=os.getenv("groq_api_key"),
            max_tokens=580,
            temperature=0
        ),
        run_config=run_config,
        bypass_n=True
    )

    embeddings = LocalEmbeddings()

    dataset = EvaluationDataset.from_list(
        create_dataset()
    )

    print("\n========== DATASET ==========\n")

    print(dataset)

    print("\n========== RUNNING RAGAS ==========\n")

    result = evaluate(
        dataset=dataset,

        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],

        llm=llm,

        embeddings=embeddings,

        run_config=run_config
    )

    print("\n========== RESULTS ==========\n")

    print(result)

    return result


if __name__ == "__main__":

    evaluate_rag()