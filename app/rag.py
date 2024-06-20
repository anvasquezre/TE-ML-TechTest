import os
from app.vector_database import query_vector_db
from dotenv import load_dotenv
from openai import OpenAI
from config.config import logger

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def retrieve_answer(question):
    logger.info(f"Retrieving answer for question: {question}")
    # Query the vector database to find relevant text snippets
    results = query_vector_db(question)

    if results is None:
        logger.error("No results returned from the vector database.")
        return "No relevant information found."

    if "documents" not in results or not results["documents"]:
        logger.error("No documents found in query results.")
        return "No relevant information found."

    # Flatten the list of lists
    documents = [doc for sublist in results["documents"] for doc in sublist]

    # Extract the relevant texts from the query results
    context = " ".join(documents)

    logger.info(f"Context for question '{question}': {context}")

    # Use OpenAI API for answer generation
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"Answer the question based on the following context: {context}",
            },
        ],
        model="gpt-4o",  # "gpt-3.5-turbo",  # or "gpt-4"
        max_tokens=150,
        temperature=1.0,
    )
    # Access the response content correctly
    answer = response.choices[0].message.content.strip()
    return answer
