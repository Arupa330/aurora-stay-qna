
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from collections import defaultdict
import os
import re


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def get_bm25_resources(vectorstore):

    # Extract all documents from the supplied FAISS vectorstore
    faiss_docs = [
        vectorstore.docstore.search(
            vectorstore.index_to_docstore_id[i]
        )
        for i in range(len(vectorstore.index_to_docstore_id))
    ]

    # Build BM25 keyword search index
    bm25_corpus = [
        tokenize(doc.page_content) for doc in faiss_docs
    ]

    bm25_retriever = BM25Okapi(bm25_corpus)

    # Map documents to their FAISS document positions
    doc_lookup = defaultdict(list)

    for i, doc in enumerate(faiss_docs):
        key = (
            doc.page_content,
            tuple(sorted(doc.metadata.items()))
        )
        doc_lookup[key].append(i)

    return faiss_docs, bm25_retriever, doc_lookup
    
def hybrid_retrieve(query, vectorstore, k=6, fetch_k=20, rrf_k=60):

    # Get BM25 resources from the same FAISS vectorstore
    faiss_docs, bm25_retriever, doc_lookup = get_bm25_resources(vectorstore)

    fetch_k = min(fetch_k, len(faiss_docs))

    # FAISS semantic search
    faiss_results = vectorstore.similarity_search(
        query, k=fetch_k
    )

    # BM25 keyword search
    bm25_scores = bm25_retriever.get_scores(tokenize(query))

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:fetch_k]

    # Reciprocal Rank Fusion
    fused_scores = defaultdict(float)

    for rank, doc in enumerate(faiss_results, start=1):
        key = (
            doc.page_content,
            tuple(sorted(doc.metadata.items()))
        )
        matches = doc_lookup.get(key, [])

        if matches:
            doc_id = matches[0]
            fused_scores[doc_id] += 1 / (rrf_k + rank)

    for rank, doc_id in enumerate(bm25_indices, start=1):
        fused_scores[doc_id] += 1 / (rrf_k + rank)

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )[:k]

    return [faiss_docs[i] for i in ranked_ids]


# ---------------------------------------------------------
# Generation
# ---------------------------------------------------------

def generate(query, retrieved_chunks, model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about AuroraStay hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {chr(10).join(
    f"Policy Chunk {i+1}:\n{doc.page_content}\nMetadata: {doc.metadata}"
    for i, doc in enumerate(retrieved_chunks)
    )}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exclusions, limitations,
   exceptions, deadlines, charges, or applicable taxes.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent hotel benefits, fees, limits, cancellation
   conditions, refund rules, occupancy limits, or policy details.
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# ---------------------------------------------------------
# Complete RAG Pipeline
# ---------------------------------------------------------

def rag(
    query: str,
    k: int = 6,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):

    if vectorstore is None:
        raise ValueError("FAISS vector store must be supplied.")

    # Initialize OpenAI LLM
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )

    # Expand cancellation-related queries
    retrieval_query = query

    if any(word in query.lower() for word in [
        "cancel", "cancellation", "refund", "money back"
    ]):
        retrieval_query += (
            " reservation cancellation policy "
            "flexible rate advance purchase non-refundable "
            "cancellation deadline refund prepayment "
            "late cancellation charge no-show"
        )

    # Hybrid retrieval: FAISS + BM25 + RRF
    retrieved_chunks = hybrid_retrieve(
        retrieval_query,
        vectorstore=vectorstore,
        k=k
    )

    # Generate answer using original question
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    if hasattr(answer, "content"):
        answer = answer.content

    return answer, retrieved_chunks
