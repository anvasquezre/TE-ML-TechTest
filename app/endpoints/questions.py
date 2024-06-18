from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from app.services.database_manager import search_similar_texts
from transformers import pipeline


router = APIRouter()


class QueryRequest(BaseModel):
    question: str


# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering")


@router.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        question = request.question
        logging.info(f"Received question: {question}")

        # Retrieve similar texts from the vector database
        collection_name = 'contract_text_embeddings'
        similar_texts = search_similar_texts(question, collection_name)
        if not similar_texts:
            raise HTTPException(status_code=404, detail="No similar texts found")

        # Combine the texts into a single context for the QA model
        context = " ".join([text["text"] for text in similar_texts])

        # Get the answer using the QA model
        answer = qa_pipeline(question=question, context=context)
        logging.info(f"Generated answer: {answer}")

        return {"question": question, "answer": answer["answer"]}
    except HTTPException as e:
        logging.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the query")

