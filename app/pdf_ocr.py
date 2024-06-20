import fitz
from config.config import logger


def extract_text_and_boxes_from_pdf(pdf_path):
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        extracted_data = []
        text_blocks = []

        # Iterate through each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]

            # Iterate through blocks
            for block in blocks:
                if block["type"] == 0:  # block type 0 means text
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            bbox = span["bbox"]
                            extracted_data.append({"text": text, "bbox": bbox})
                            text_blocks.append(text)

        return extracted_data, text_blocks
    except Exception as e:
        logger.error(f"Error extracting text and bounding boxes from PDF: {e}")
        return [], []
