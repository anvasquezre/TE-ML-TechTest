import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2


def display_extracted_data(img_path, json_path):
    try:
        # Load the image using OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load the extracted data from JSON file
        with open(json_path, "r") as f:
            extracted_data = json.load(f)

        # Plot the image and add bounding boxes
        _, ax = plt.subplots(1)
        ax.imshow(image)

        for data in extracted_data:
            text = data["text"]
            bbox = data["bbox"]
            # Convert relative bbox to absolute coordinates
            (x0, y0), (x1, y1) = bbox
            x0, y0 = int(x0 * image.shape[1]), int(y0 * image.shape[0])
            x1, y1 = int(x1 * image.shape[1]), int(y1 * image.shape[0])
            width, height = x1 - x0, y1 - y0
            # Draw rectangle
            rect = patches.Rectangle(
                (x0, y0), width, height, linewidth=1, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x0, y0, text, bbox=dict(facecolor="yellow", alpha=0.5))

        plt.show()
    except Exception as e:
        print(f"Error displaying extracted data: {e}")


if __name__ == "__main__":
    img_path = "data/driver_license_data/rotated_360 (1).png"
    json_path = "data/extracted_data.json"
    display_extracted_data(img_path, json_path)
