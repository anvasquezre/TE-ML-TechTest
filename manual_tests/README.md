# Manual Tests

This folder contains scripts for manually testing various parts of the system. Each script runs a specific test and prints the results to the console for verification.

## Test Scripts

1. **run_ocr.py**: Tests the OCR functionality to extract text and bounding boxes from the driver's license image.
2. **run_image_alignment.py**: Aligns the rotated driver's license image and saves the result.
3. **run_ner.py**: Tests the Named Entity Recognition (NER) functionality on the extracted text.
4. **run_bounding_boxes.py**: Matches extracted names with their bounding box coordinates.
5. **run_api_tests.py**: Tests the API endpoints by sending requests and printing the responses.

## Running the Tests

To run a test, navigate to the `manual_tests` folder and execute the script using Python. For example:

- To run the OCR test:
    ```sh
    python run_ocr.py
    ```

Make sure the FastAPI server is running before testing the API endpoints. You can start the server using:

- To start the server:
    ```sh
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

## Requirements

- Python 3.9+
- Tesseract OCR
- OpenCV
- Requests library

Install the required libraries using:

- To install the libraries:
    ```sh
    pip install pytesseract opencv-python requests
    ```

### Summary

- **Manual Test Scripts**: We created separate scripts for manually testing each functionality.
- **README**: Provides instructions on running the manual tests and what each script does.

These manual tests will help you interactively verify the functionality of your system and ensure everything works as expected. Let me know if you need any further assistance!
