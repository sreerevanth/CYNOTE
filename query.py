from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

DB_PATH = "embeddings"


def ask_question(question: str):
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load vector database
    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve relevant chunks
    docs = db.similarity_search(question, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    # Local LLM
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    prompt = f"""
    You are a study assistant.

    Using ONLY the context below:
    - Explain the concept in detail
    - Include definition, purpose, and importance
    - Write in simple academic language
    - Do not add information outside the context

    Context:
    {context}

    Question:
    {question}
    """


    answer = llm(prompt)[0]["generated_text"]

    return answer, docs


if __name__ == "__main__":
    question = "Summarize the main ideas from the document."
    answer, sources = ask_question(question)

    print("\nANSWER:\n")
    print(answer)

    print("\nSOURCES:")
    for i, doc in enumerate(sources, 1):
        print(f"{i}.", doc.metadata)
