import json
from app.pdf_ocr import extract_text_and_boxes_from_pdf
from app.vector_database import store_text_in_vector_db
from config.config import logger

if __name__ == "__main__":
    pdf_file_name = "AS-IS-Residential-Contract-for-Sale-And-Purchase (1).pdf"
    pdf_path = f"data/residential_contract/{pdf_file_name}"
    extracted_data, text_blocks = extract_text_and_boxes_from_pdf(pdf_path)

    logger.info(f"Extracted Data: {extracted_data}")
    logger.info(f"Text Blocks: {text_blocks}")

    # Save the results to a JSON file
    output_path = "output.json"
    with open(output_path, "w") as f:
        json.dump(
            {"extracted_data": extracted_data, "text_blocks": text_blocks}, f, indent=4
        )

    logger.info(f"Extraction completed. Results saved to {output_path}")

    # Store the extracted text in the vector database
    for i, text in enumerate(text_blocks):
        doc_id = f"{pdf_file_name}_page_{i+1}"
        store_text_in_vector_db(doc_id, text)
        logger.info(f"Stored text from page {i+1} in vector database.")
