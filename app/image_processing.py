import cv2
import numpy as np
import re
from pytesseract import image_to_osd
from PIL import Image
from config.config import logger


def get_image_orientation(image_path):
    try:
        image = Image.open(image_path)
        osd = image_to_osd(image)
        angle = int(re.search(r"(?<=Rotate: )\d+", osd).group(0))
        if angle == 90:
            angle = -90
        elif angle == 270:
            angle = -270
        logger.info(f"Detected angle from OSD: {angle} degrees")
        return angle
    except Exception as e:
        logger.error(f"Error detecting orientation: {e}")
        return 0


def detect_angle_with_contours(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(
            edged, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )

        if lines is not None:
            angles = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)

            median_angle = np.median(angles)

            if median_angle < -45:
                median_angle += 90
            elif median_angle > 45:
                median_angle -= 90

            logger.info(
                f"Detected angle from Hough Line Transform: {median_angle} degrees"
            )
            return median_angle
        else:
            logger.info("No lines detected")
            return 0
    except Exception as e:
        logger.error(f"Error detecting angle with contours: {e}")
        return 0


def rotate_image(image, angle):
    try:
        if angle != 0:
            (h, w) = image.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        else:
            return image
    except Exception as e:
        logger.error(f"Error rotating image: {e}")
        return None


def save_image(image, output_path):
    try:
        cv2.imwrite(output_path, image)
        logger.info(f"Image saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")


def fix_image_orientation(image_path, output_path):
    try:
        logger.info(f"Loading image from {image_path}")
        image = cv2.imread(image_path)

        # Try to get orientation from OSD first
        angle = get_image_orientation(image_path)
        if angle == 0:
            # If OSD fails to provide an angle, use contour analysis
            angle = detect_angle_with_contours(image)

        if angle != 0:
            rotated_image = rotate_image(image, angle)  # Rotate by angle to correct
            if rotated_image is not None:
                save_image(rotated_image, output_path)
                logger.info(f"Image saved to {output_path}")
            else:
                logger.error("Rotated image is None")
        else:
            logger.info("No rotation needed for the image")
            save_image(image, output_path)
    except Exception as e:
        logger.error(f"Error fixing image orientation: {e}")
