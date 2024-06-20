from PIL import Image
import pytesseract
from pytesseract import Output
from config.config import logger
from app.ner import extract_names


def extract_text_and_boxes(img_path):
    try:
        text = pytesseract.image_to_string(Image.open(img_path))
        # logger.info(f"OCR Extracted Text: {text}")

        bounding_boxes = pytesseract.image_to_data(
            Image.open(img_path), output_type=Output.DICT
        )
        # logger.info(f"OCR Extracted Bounding Boxes: {bounding_boxes}")
        # for i in range(len(bounding_boxes["level"])):
        #     logger.info(
        #         f"Box {i}: ({bounding_boxes['left'][i]}, {bounding_boxes['top'][i]}, {bounding_boxes['width'][i]}, {bounding_boxes['height'][i]}) - {bounding_boxes['text'][i]}"
        #     )
        return text
    except Exception as e:
        logger.error(f"Error extracting text or bounding boxes: {e}")
        return None


def preprocess_text(text):
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


def test_extract_names(img_path):
    try:
        text = extract_text_and_boxes(img_path)
        if text:
            text = preprocess_text(text)
            logger.info(f"Preprocessed OCR Text: {text}")
            names = extract_names(text)
            logger.info(f"Extracted Names: {names}")
    except Exception as e:
        logger.error(f"Error extracting names: {e}")


if __name__ == "__main__":
    file_name = "corrected_image_rotated_30 (1).png"
    img_path = f"data/driver_license_data/{file_name}"
    test_extract_names(img_path)
