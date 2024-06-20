import os
from doctr.io import DocumentFile
from doctr.models import db_resnet50, crnn_vgg16_bn
from config.config import logger
from app.doctr_module import (
    extract_text_and_boxes_doctr,
    extract_names_and_boxes,
    load_custom_model,
)

if __name__ == "__main__":
    # img_path = "data/driver_license_data/rotated_360 (1).png"
    img_path = "data/driver_license_data/rotated_30 (1).png"
    det_weights_path = "models/db_resnet50-79bd7d70.pt"
    reco_weights_path = "models/crnn_vgg16_bn-9762b0b0.pt"

    # Load custom models
    det_model, reco_model = load_custom_model(
        db_resnet50, det_weights_path, crnn_vgg16_bn, reco_weights_path
    )
    # extract_text_and_boxes_doctr(img_path, det_weights_path, reco_weights_path)
    extract_names_and_boxes(img_path, det_model, reco_model)
