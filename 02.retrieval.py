import logging
import os
import re
import sys
import threading
from pathlib import Path

import dspy
import dspy.streaming
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationRailsOptions
from huggingface_hub import InferenceClient
from pinecone import Pinecone
from pypdf import PdfReader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


_BASE_DIR = Path(__file__).resolve().parent
_DATA_PATH = _BASE_DIR / "data" / "Attntion.pdf"
logger = logging.getLogger(__name__)

lm = None
index = None
client = None
bm25_retriever = None
_pipeline_lock = threading.Lock()
_configure_lock = threading.Lock()
_bm25_lock = threading.Lock()
_guardrails_lock = threading.Lock()
_guardrails = None

PINECONE_API_KEY = None
GROQ_API_KEY = None
HF_TOKEN = None
GUARDRAIL_REFUSAL = "I'm sorry, I can't respond to that."
MAX_QUERY_LENGTH = 2000
MAX_CONVERSATION_HISTORY_LENGTH = 12000
CASUAL_GREETING_RESPONSES = {
    "hi": "Hello! How can I help?",
    "hello": "Hello! How can I help?",
    "hey": "Hey! How can I help?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
}


class RetrievalServiceError(RuntimeError):
    pass


class RetrievalResults(list):
    def __init__(self, results=(), direct_response=""):
        super().__init__(results)
        self.direct_response = direct_response


def _load_deployment_secrets():
    load_dotenv()
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        secrets = {}

    for environment_name, secret_names in {
        "GROQ_API_KEY": ("GROQ_API_KEY", "groq_api_key"),
        "PINECONE_API_KEY": ("PINECONE_API_KEY", "pinecone_api_key"),
        "HF_TOKEN": ("HF_TOKEN", "hf_token"),
    }.items():
        if os.getenv(environment_name):
            continue
        for secret_name in secret_names:
            value = secrets.get(secret_name)
            if value:
                os.environ[environment_name] = str(value)
                break


def _guardrail_content(response) -> str:
    if isinstance(response, dict):
        return str(response.get("content", ""))
    return str(getattr(response, "content", ""))


def _load_guardrails():
    global _guardrails
    if _guardrails is not None:
        return _guardrails

    with _guardrails_lock:
        if _guardrails is None:
            _load_deployment_secrets()
            config = RailsConfig.from_path(str(_BASE_DIR / "guardrails"))
            _guardrails = LLMRails(config)
    return _guardrails


def check_message(message: str, role: str) -> str:
    _validate_query(message)
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")

    options = GenerationOptions(
        rails=GenerationRailsOptions(
            input=["self check input"] if role == "user" else False,
            output=["self check output"] if role == "assistant" else False,
            retrieval=False,
            dialog=False,
            tool_input=False,
            tool_output=False,
        ),
        llm_params={"max_tokens": 16, "temperature": 0},
    )
    response = _load_guardrails().generate(
        messages=[{"role": role, "content": message}],
        options=options,
    )
    return _guardrail_content(response)


def _grounded_answer(answer: str, context: str) -> bool:
    stop_words = {
        "about", "after", "again", "also", "because", "being", "between",
        "could", "does", "from", "have", "into", "more", "only", "other",
        "should", "some", "than", "that", "their", "there", "these", "they",
        "this", "through", "using", "what", "when", "where", "which", "with",
        "would", "your",
    }
    normalize = lambda value: re.findall(r"[a-z0-9]+", value.lower())
    answer_terms = {
        word
        for word in normalize(answer)
        if len(word) >= 4 and word not in stop_words
    }
    context_terms = set(normalize(context))
    return bool(answer_terms & context_terms)


def _validate_query(query: str):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"query must not exceed {MAX_QUERY_LENGTH} characters"
        )


def _validate_conversation_history(conversation_history: str):
    if not isinstance(conversation_history, str):
        raise ValueError("conversation_history must be a string")
    if len(conversation_history) > MAX_CONVERSATION_HISTORY_LENGTH:
        raise ValueError(
            "conversation_history exceeds the maximum allowed length"
        )


def _validate_top_k(top_k: int):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def _validate_chunking(chunk_size: int, overlap: int):
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(overlap, int) or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be an integer from 0 to chunk_size - 1")


def _validate_environment():
    missing = [
        name
        for name, value in (
            ("pinecone_api_key", PINECONE_API_KEY),
            ("groq_api_key", GROQ_API_KEY),
            ("hf_token", HF_TOKEN),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def configure_dspy():
    """Configure DSPy once, from whichever thread initializes the pipeline first."""
    if dspy.settings.lm is not None:
        return

    with _configure_lock:
        if dspy.settings.lm is not None:
            return

        try:
            dspy.configure(lm=lm, verbose=True)
        except RuntimeError:
            if dspy.settings.lm is None:
                raise


def initialize_pipeline():
    """Initialize external clients and local retrieval data once."""
    global PINECONE_API_KEY, GROQ_API_KEY, HF_TOKEN
    global lm, index, client, bm25_retriever

    if bm25_retriever is not None:
        return

    with _pipeline_lock:
        if bm25_retriever is not None:
            return

        _load_deployment_secrets()
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or os.getenv("pinecone_api_key")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
        HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("hf_token")
        _validate_environment()

        lm = dspy.LM(
            "groq/qwen/qwen3.8-27b",
            api_key=GROQ_API_KEY,
            max_tokens=512,
            temperature=0,
            timeout=45,
            num_retries=1,
        )
        configure_dspy()
        index = Pinecone(api_key=PINECONE_API_KEY).Index("attntion-384")
        client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

        pdf_text = load_pdf()
        chunks = create_chunks(pdf_text, chunk_size=1000, overlap=200)
        docs = create_documents(chunks)
        bm25_retriever = create_bm25_retriever(docs, k=5)


class QueryReWrite(dspy.Signature):
    """
    Classify the message before retrieval.

    Use intent=document_question only when the message asks about the indexed
    document or its attention mechanism. Use intent=direct_response for
    greetings, thanks, casual conversation, or unrelated questions.
    For direct_response, answer briefly and do not create a search query.
    For document_question, do not answer; rewrite the query for retrieval.
    """

    query: str = dspy.InputField(
        description="The user's original message"
    )

    conversation_history: str = dspy.InputField(
        description="The conversation history, if any"
    )

    intent: str = dspy.OutputField(
        description="Exactly document_question or direct_response"
    )

    re_written_query: str = dspy.OutputField(
        description="A clearer search query, or empty for direct_response"
    )

    direct_response: str = dspy.OutputField(
        description="A brief direct reply, or empty for document_question"
    )

    


query_re_writer = dspy.Predict(QueryReWrite)


class AnswerQuestion(dspy.Signature):
    """
    Answer the user's question using only the provided context.

    Base the answer strictly on the context.
    If it is general answer, provide a concise and accurate response.
    And tell if the answer is not in the context.
    """

    context: str = dspy.InputField(
        description="Retrieved passages relevant to the question"
    )

    question: str = dspy.InputField(
        description="The user's original question"
    )

    conversation_history: str = dspy.InputField(
        description="The conversation history, if any"
    )

    answer: str = dspy.OutputField(
        description="An answer grounded in the provided context"
    )


answer_generator = dspy.Predict(AnswerQuestion)


class HealthCheck(dspy.Signature):
    prompt: str = dspy.InputField()
    response: str = dspy.OutputField()


health_checker = dspy.Predict(HealthCheck)


# ============================================================
# 4. HUGGING FACE EMBEDDING MODEL
# ============================================================

def generate_embedding(text: str):
    initialize_pipeline()

    try:
        embedding = client.feature_extraction(
            text,
            model="BAAI/bge-small-en-v1.5"
        )
    except Exception as exc:
        raise RetrievalServiceError(
            "Hugging Face embedding request failed"
        ) from exc

    return embedding.tolist()


# ============================================================
# 5. LOAD PDF
# ============================================================

def load_pdf():

    reader = PdfReader(_DATA_PATH)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# 6. CHUNK PDF
# ============================================================

def create_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    _validate_chunking(chunk_size, overlap)

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(
        0,
        len(words),
        step
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk:
            chunks.append(chunk)

    return chunks


# ============================================================
# 7. KEYWORD SEARCH
# ============================================================

def create_documents(chunks):

    docs = []

    for i, chunk in enumerate(chunks):

        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "chunk_id": f"chunk_{i}"
                }
            )
        )

    return docs

def create_bm25_retriever(
    docs,
    k: int = 5
):
    _validate_top_k(k)

    retriever = BM25Retriever.from_documents(
        docs
    )

    retriever.k = k

    return retriever


# ============================================================
# 8. GET KEYWORD RESULTS
# ============================================================
def get_chunks_by_keywords(
    query: str,
    top_k: int = 5
):
    _validate_query(query)
    _validate_top_k(top_k)
    initialize_pipeline()

    with _bm25_lock:
        bm25_retriever.k = top_k
        results = bm25_retriever.invoke(query)

    keyword_results = []

    for rank, doc in enumerate(
        results,
        start=1
    ):

        keyword_results.append({
            "chunk_id": doc.metadata["chunk_id"],
            "chunk": doc.page_content,
            "rank": rank
        })

    return keyword_results

# ============================================================
# 9. PINECONE SIMILARITY SEARCH
# ============================================================

def get_chunks_by_similarity(
    query_embedding,
    top_k: int = 5
):
    if not query_embedding:
        raise ValueError("query_embedding must not be empty")
    _validate_top_k(top_k)
    initialize_pipeline()

    similarity_results = []


    try:
        result = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
    except Exception as exc:
        raise RetrievalServiceError(
            "Pinecone similarity search failed"
        ) from exc

    for rank, match in enumerate(
            result.matches,
            start=1
        ):
        metadata = match.metadata or {}
        chunk = metadata.get("text") if isinstance(metadata, dict) else None
        if not isinstance(chunk, str) or not chunk.strip():
            continue
        if not isinstance(match.id, str) or not match.id.strip():
            continue

        similarity_results.append({
            "chunk_id": match.id,
            "chunk": chunk,
            "similarity_score": match.score,
            "rank": rank
        })

    return similarity_results

# ============================================================
# 10. COMBINE RESULTS
# ============================================================

def combine_results(
    similarity_results,
    keyword_results,
    top_k: int = 5,
    rrf_k: int = 60
):
    _validate_top_k(top_k)
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer")

    combined = {}

    # --------------------------------
    # Similarity results
    # --------------------------------

    for result in similarity_results:

        chunk_id = result["chunk_id"]

        if chunk_id not in combined:
            
            combined[chunk_id] = {
                 "chunk_id": chunk_id,
                 "chunk": result["chunk"],
                 "semantic_rank": None,
                 "keyword_rank": None,
                 "similarity_score": 0,
                  "rrf_score": 0            
                }

            combined[chunk_id]["semantic_rank"] = (
                result["rank"]
            )

            combined[chunk_id]["similarity_score"] = (
                result["similarity_score"]
            )
            combined[chunk_id]["rrf_score"] += (
                        1 / (rrf_k + result["rank"])
                    )


    for result in keyword_results:

        chunk_id = result["chunk_id"]

        if chunk_id not in combined:

            combined[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk": result["chunk"],
                "semantic_rank": None,
                "keyword_rank": None,
                "similarity_score": 0,
                "rrf_score": 0
            }

        combined[chunk_id]["keyword_rank"] = (
            result["rank"]
        )

        combined[chunk_id]["rrf_score"] += (
            1 / (rrf_k + result["rank"])
        )

    results = sorted(
        combined.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# 11. HEALTH CHECK
# ============================================================

def health_check():
    status = {}
    try:
        initialize_pipeline()
    except Exception as exc:
        status["initialization"] = {"ok": False, "error": str(exc)}
        status["ok"] = False
        return status

    status["pdf"] = {
        "ok": _DATA_PATH.is_file() and _DATA_PATH.stat().st_size > 0
    }
    status["bm25"] = {
        "ok": bm25_retriever is not None
    }

    try:
        index.describe_index_stats()
        status["pinecone"] = {"ok": True}
    except Exception as exc:
        status["pinecone"] = {"ok": False, "error": str(exc)}

    try:
        embedding = client.feature_extraction(
            "health check",
            model="BAAI/bge-small-en-v1.5"
        )
        status["hugging_face"] = {"ok": embedding is not None}
    except Exception as exc:
        status["hugging_face"] = {"ok": False, "error": str(exc)}

    try:
        with dspy.context(lm=lm):
            health_checker(prompt="Reply with OK")
        status["groq"] = {"ok": True}
    except Exception as exc:
        status["groq"] = {"ok": False, "error": str(exc)}

    status["ok"] = all(
        component["ok"]
        for name, component in status.items()
        if name != "ok"
    )
    return status


# ============================================================
# 12. COMPLETE RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    conversation_history: str = ""
):
    _validate_query(query)
    _validate_top_k(top_k)
    _validate_conversation_history(conversation_history)

    direct_response = CASUAL_GREETING_RESPONSES.get(query.strip().lower())
    if direct_response:
        return RetrievalResults(direct_response=direct_response)

    try:
        input_check = check_message(query, "user")
    except Exception as exc:
        raise RetrievalServiceError("Input safety check failed") from exc
    if input_check.strip().lower() == GUARDRAIL_REFUSAL.lower():
        return RetrievalResults(direct_response=GUARDRAIL_REFUSAL)

    initialize_pipeline()

    # --------------------------------
    # Step 1: Query rewriting
    # --------------------------------

    try:
        with dspy.context(lm=lm):
            rewritten_result = query_re_writer(
                query=query,
                conversation_history=conversation_history
            )
    except Exception as exc:
        raise RetrievalServiceError(
            "Groq query rewriting request failed"
        ) from exc

    intent = str(getattr(rewritten_result, "intent", "")).strip().lower()
    if intent == "direct_response":
        direct_response = str(
            getattr(rewritten_result, "direct_response", "")
        ).strip()
        if not direct_response:
            raise RetrievalServiceError(
                "Query rewrite returned an empty direct response"
            )
        return RetrievalResults(direct_response=direct_response)

    if intent != "document_question":
        raise RetrievalServiceError(
            "Query rewrite returned an invalid intent"
        )

    rewritten_query = str(
        getattr(rewritten_result, "re_written_query", "")
    ).strip()
    _validate_query(rewritten_query)

    # --------------------------------
    # Step 2: Semantic search
    # Use rewritten query
    # --------------------------------

    query_embedding = generate_embedding(
        rewritten_query
    )

    similarity_results = (
        get_chunks_by_similarity(
            query_embedding,
            top_k=top_k
        )
    )

    # --------------------------------
    # Step 3: Keyword search
    # Use ORIGINAL query
    # --------------------------------

    keyword_results = (
        get_chunks_by_keywords(
            query
        )
    )

    # --------------------------------
    # Step 4: Combine
    # --------------------------------

    results = combine_results(
        similarity_results,
        keyword_results
    )

    return RetrievalResults(results[:top_k])


# ============================================================
# 12. GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str,
    results,
    conversation_history: str = ""
):
    _validate_query(query)
    if not isinstance(results, (list, tuple)):
        raise ValueError("results must be a list or tuple")
    _validate_conversation_history(conversation_history)
    initialize_pipeline()

    context = "\n\n".join(
        result["chunk"]
        for result in results
    )

    with dspy.context(lm=lm):
        response = answer_generator(
            context=context,
            question=query,
            conversation_history=conversation_history
        )

    answer = response.answer
    if context.strip() and not _grounded_answer(answer, context):
        return "I could not find that information in the retrieved context."

    try:
        output_check = check_message(answer, "assistant")
    except Exception as exc:
        raise RetrievalServiceError("Output safety check failed") from exc
    if output_check.strip().lower() == GUARDRAIL_REFUSAL.lower():
        return GUARDRAIL_REFUSAL

    return answer


# ============================================================
# 12b. STREAM ANSWER
# ============================================================

answer_streamer = dspy.streamify(
    answer_generator,
    stream_listeners=[
        dspy.streaming.StreamListener(
            signature_field_name="answer"
        )
    ],
    async_streaming=False,
)


def stream_answer(
    query: str,
    results,
    conversation_history: str = ""
):
    _validate_query(query)
    if not isinstance(results, (list, tuple)):
        raise ValueError("results must be a list or tuple")
    _validate_conversation_history(conversation_history)
    initialize_pipeline()

    context = "\n\n".join(
        result["chunk"]
        for result in results
    )

    streamed = False

    final_prediction = None

    for value in answer_streamer(
        context=context,
        question=query,
        conversation_history=conversation_history
    ):

        if isinstance(
            value,
            dspy.streaming.StreamResponse
        ):
            streamed = True
            yield value.chunk

        elif isinstance(
            value,
            dspy.Prediction
        ):
            final_prediction = value

    if (
        not streamed
        and final_prediction is not None
        and getattr(
            final_prediction,
            "answer",
            None
        )
    ):
        yield final_prediction.answer


# ============================================================
# 13. MAIN
# ============================================================

def main():

    query = (
        "What is the attention mechanism "
        "and how does it work?"
    )

    results = retrieve_relevant_chunks(
        query,
        top_k=5
    )

    answer = generate_answer(
        query,
        results
    )

    logging.basicConfig(level=logging.INFO)
    logger.info("========== ANSWER ==========")
    logger.info(answer)

 
 


if __name__ == "__main__":
    main()