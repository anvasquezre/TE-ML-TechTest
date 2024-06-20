import os
from app.image_processing import fix_image_orientation
from config.config import logger

if __name__ == "__main__":
    file_name = "rotated_30 (1).png"  # for example, this is the file name for the 30 degrees rotation
    # file_name = "rotated_360 (1).png"  # for example, this is the file name for the 360 degrees rotation
    logger.info(f"Starting orientation fix for data/driver_license_data/{file_name}")
    image_path = f"data/driver_license_data/{file_name}"
    output_path = f"data/driver_license_data/corrected_image_{file_name}"
    fix_image_orientation(image_path, output_path)
    logger.info(f"Completed orientation fix for data/driver_license_data/{file_name}")
