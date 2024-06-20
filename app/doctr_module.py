from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import re
import torch
from config.config import logger
from app.ner import extract_names
from app.image_processing import fix_image_orientation


def load_custom_model(
    det_model_class, det_weights_path, reco_model_class, reco_weights_path
):
    # Load custom detection model
    det_model = det_model_class(pretrained=False, pretrained_backbone=False)
    det_params = torch.load(det_weights_path, map_location="cpu")
    det_model.load_state_dict(det_params)
    logger.info(f"Loaded detection model from {det_weights_path}")

    # Load custom recognition model
    reco_model = reco_model_class(pretrained=False, pretrained_backbone=False)
    reco_params = torch.load(reco_weights_path, map_location="cpu")
    reco_model.load_state_dict(reco_params)
    logger.info(f"Loaded recognition model from {reco_weights_path}")

    return det_model, reco_model


def extract_text_and_boxes_doctr(img_path, det_model, reco_model):
    try:
        # Fix image orientation
        output_path = "temp_fixed_image.png"  # Temporary path for the fixed image
        fix_image_orientation(img_path, output_path)

        doc = DocumentFile.from_images(output_path)

        # Initialize the OCR predictor with local models
        predictor = ocr_predictor(
            det_arch=det_model, reco_arch=reco_model, pretrained=False
        )
        logger.info("OCR predictor initialized with custom models.")

        # Perform OCR
        result = predictor(doc)

        extracted_data = []
        text_blocks = []

        # Extract text and bounding boxes
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text = word.value
                        bbox = word.geometry
                        extracted_data.append({"text": text, "bbox": bbox})
                        text_blocks.append(text)
                        # logger.info(f"Extracted text: {text}, Bounding box: {bbox}")

        return extracted_data, text_blocks
    except Exception as e:
        logger.error(f"Error extracting text or bounding boxes with DocTR: {e}")
        return [], []


def preprocess_text(text_blocks):
    text = " ".join(text_blocks)
    clean_text = re.sub(r"[^a-zA-Z\s]", "", text).strip().lower()
    return clean_text


def extract_names_and_boxes(img_path, det_model, reco_model):
    try:
        # Extract text and bounding boxes using DocTR
        extracted_data, text_blocks = extract_text_and_boxes_doctr(
            img_path, det_model, reco_model
        )

        # Check if any text blocks were extracted
        if text_blocks:
            # Preprocess the text blocks for NER
            preprocessed_text = preprocess_text(text_blocks)
            logger.info(f"Preprocessed OCR Text: {preprocessed_text}")

            # Extract names using the NER model
            names = extract_names(preprocessed_text)
            logger.info(f"Extracted Names: {names}")

            # Split the extracted names into a list of individual names
            name_parts = re.findall(r"\b\w+\b", names)
            logger.info(f"Name Parts: {name_parts}")

            # Initialize a list to hold the names and their bounding boxes
            name_boxes = []

            # Match each name part with its bounding box
            for data in extracted_data:
                for part in name_parts:
                    if part.lower() in data["text"].lower():
                        box = {
                            "name": part,
                            "left": data["bbox"][0][0],
                            "top": data["bbox"][0][1],
                            "right": data["bbox"][1][0],
                            "bottom": data["bbox"][1][1],
                        }
                        name_boxes.append(box)
                        logger.info(f"Matched part: {part} with text: {data['text']}")

            # Combine bounding boxes for full names
            combined_boxes = []
            full_name = names.strip()
            full_name_parts = full_name.split()

            # Adjusted part to return bbox as a nested dictionary
            if len(full_name_parts) > 1:
                boxes_for_full_name = [
                    box for box in name_boxes if box["name"] in full_name_parts
                ]
                if boxes_for_full_name:
                    combined_box = {
                        "name": full_name,
                        "bbox": {
                            "left": min(box["left"] for box in boxes_for_full_name),
                            "top": min(box["top"] for box in boxes_for_full_name),
                            "right": max(box["right"] for box in boxes_for_full_name),
                            "bottom": max(box["bottom"] for box in boxes_for_full_name),
                        },
                    }
                    combined_boxes.append(combined_box)
                    logger.info(f"Combined box for {full_name}: {combined_box}")

            if combined_boxes:
                for combined_box in combined_boxes:
                    logger.info(
                        f"Combined Name: {combined_box['name']}, Bounding box: {combined_box}"
                    )
            else:
                logger.info("No names were matched with bounding boxes.")

            return (
                combined_boxes,
                text_blocks,
            )  # Return the combined names and their bounding boxes along with the text blocks

        else:
            logger.info("No text blocks extracted from the image.")
            return [], []  # Return empty lists if no text blocks were found

    except Exception as e:
        logger.error(f"Error in NER process: {e}")
        return [], []  # Return empty lists in case of an error
