import json
from app.rag import retrieve_answer
from config.config import logger


if __name__ == "__main__":

    sample_question = "What are the financing options available in this contract?"
    # sample_question = "What are the buyer's obligations in the contract?"
    # sample_question = "What is the deadline for the buyer to apply for financing?"
    # sample_question = "What are the conditions for loan approval in the contract?"

    # Retrieve the answer using the RAG approach
    answer = retrieve_answer(sample_question)
    logger.info(f"Generated Answer: {answer}")

    # Save the answer to a JSON file
    output_path = "data/results/answer_output.json"
    with open(output_path, "w") as f:
        json.dump({"question": sample_question, "answer": answer}, f, indent=4)

    logger.info(f"Answer saved to {output_path}")
