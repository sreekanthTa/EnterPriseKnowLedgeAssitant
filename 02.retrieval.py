import os
import sys
import threading

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import dspy
import dspy.streaming
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pinecone import Pinecone
from pypdf import PdfReader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("pinecone_api_key")
GROQ_API_KEY = os.getenv("groq_api_key")
HF_TOKEN = os.getenv("hf_token")


# ============================================================
# 2. PINECONE
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index("attntion-384")


# ============================================================
# 3. DSPY + GROQ
# ============================================================

lm = dspy.LM(
    "groq/qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY,
    max_tokens=1024,
    temperature=0,
    timeout=45,
    num_retries=1
)

_configure_lock = threading.Lock()


def configure_dspy():
    """Configure DSPy once, from whichever thread imports this module first.

    DSPy only lets the thread that called ``dspy.configure`` call it again.
    Streamlit reruns the script in a fresh thread each time (and re-executes
    this module whenever the file changes invalidate ``st.cache_resource``),
    so a second unconditional call raises ``RuntimeError``. Reading the global
    settings is allowed from any thread, so skip the write when already set.
    """
    if dspy.settings.lm is not None:
        return

    with _configure_lock:
        if dspy.settings.lm is not None:
            return

        try:
            dspy.configure(
                lm=lm,
                verbose=True
            )
        except RuntimeError:
            if dspy.settings.lm is None:
                raise


configure_dspy()


class QueryReWrite(dspy.Signature):
    """
    Rewrite the user's query for semantic retrieval.

    Preserve the original meaning and important context.
    Make the query clear, specific and focused.
    Do not answer the question.
    """

    query: str = dspy.InputField(
        description="The user's original question"
    )

    re_written_query: str = dspy.OutputField(
        description="A clearer and more specific search query"
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

    answer: str = dspy.OutputField(
        description="An answer grounded in the provided context"
    )


answer_generator = dspy.Predict(AnswerQuestion)


# ============================================================
# 4. HUGGING FACE EMBEDDING MODEL
# ============================================================

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)


def generate_embedding(text: str):

    embedding = client.feature_extraction(
        text,
        model="BAAI/bge-small-en-v1.5"
    )

    return embedding.tolist()


# ============================================================
# 5. LOAD PDF
# ============================================================

def load_pdf():

    reader = PdfReader(
        "data/Attntion.pdf"
    )

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

    retriever = BM25Retriever.from_documents(
        docs
    )

    retriever.k = k

    return retriever

pdf_text = load_pdf()

chunks = create_chunks(
    pdf_text,
    chunk_size=1000,
    overlap=200
)

docs = create_documents(
    chunks
)

bm25_retriever = create_bm25_retriever(
    docs,
    k=5
)

 
# ============================================================
# 8. GET KEYWORD RESULTS
# ============================================================
def get_chunks_by_keywords(
    query: str,
    top_k: int = 5
):

    bm25_retriever.k = top_k

    results = bm25_retriever.invoke(
        query
    )

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

    similarity_results = []


    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    for rank, match in enumerate(
            result.matches,
            start=1
        ):

            similarity_results.append({
                "chunk_id": match.id,
                "chunk": match.metadata["text"],
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
# 11. COMPLETE RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5
):

    # --------------------------------
    # Step 1: Query rewriting
    # --------------------------------

    with dspy.context(lm=lm):
        rewritten_result = query_re_writer(
            query=query
        )

    rewritten_query = (
        rewritten_result.re_written_query
    )

    print(
        "\nRe-written Query:",
        rewritten_query
    )

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

    return results[:top_k]


# ============================================================
# 12. GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str,
    results
):

    context = "\n\n".join(
        result["chunk"]
        for result in results
    )

    with dspy.context(lm=lm):
        response = answer_generator(
            context=context,
            question=query
        )

    return response.answer


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
    results
):

    context = "\n\n".join(
        result["chunk"]
        for result in results
    )

    streamed = False

    final_prediction = None

    for value in answer_streamer(
        context=context,
        question=query
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

    print("\n\n========== ANSWER ==========\n")

    print(answer)

 
 


if __name__ == "__main__":
    main()