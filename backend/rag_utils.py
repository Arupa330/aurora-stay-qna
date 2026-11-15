
from langchain_groq import ChatGroq
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function
def generate(query, retrieved_chunks, model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about AuroraStay hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exclusions, limitations, exceptions, or charges.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent benefits, fees, limits, conditions, or policy details.
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline
def rag(
    query: str,
    k: int = 2,
    model_name: str = "openai/gpt-oss-20b",
    temperature: float = 0.0,
    max_tokens: int = 500,
    vectorstore=None
):
    # Create the Groq generator model
    model = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore
    )

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
