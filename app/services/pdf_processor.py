import fitz
from transformers import pipeline
from fuzzywuzzy import fuzz
import json


def extract_text_and_coordinates_by_line(pdf_path):
    """
    Extracts text and coordinates from the given PDF by line using PyMuPDF.

    Parameters:
    pdf_path (str): The path to the PDF file.

    Returns:
    list: A list of dictionaries, each containing the line text and its coordinates.
    """
    document = fitz.open(pdf_path)
    lines_with_coords = []

    for page_num in range(len(document)):
        page = document[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    line_coords = []
                    for span in line["spans"]:
                        line_text += span["text"] + " "
                        line_coords.append((span["bbox"], span["text"]))
                    lines_with_coords.append({
                        "text": line_text.strip(),
                        "coords": line_coords,
                        "page": page_num
                    })

    return lines_with_coords


def extract_names(text):
    """
    Extracts names from the given text using a Named Entity Recognition (NER) model.

    Parameters:
    text (str): The input text from which to extract names.

    Returns:
    list:  A list of extracted names.
    """
    ner_pipeline = pipeline("ner",
                            model="dbmdz/bert-large-cased-finetuned-conll03-english",
                            framework="pt")
    ner_results = ner_pipeline(text)
    names = []
    current_name = []
    previous_end = -1

    for entity in ner_results:
        if entity['entity'] in ['B-PER', 'I-PER']:
            if entity['word'].startswith("##"):
                current_name[-1] += entity['word'][2:]
            else:
                if current_name and (entity['start'] - previous_end) > 1:
                    names.append(" ".join(current_name))
                    current_name = []
                current_name.append(entity['word'])
            previous_end = entity['end']
        else:
            if current_name:
                names.append(" ".join(current_name))
                current_name = []
            previous_end = -1

    if current_name:
        names.append(" ".join(current_name))

    return names


def find_name_coordinates(pdf_path, name):
    """
    Finds the coordinates of the specified name in the given PDF using PyMuPDF.

    Parameters:
    pdf_path (str): The path to the PDF file.
    name (str): The name for which to find the coordinates.

    Returns:
    dict: A dictionary containing the name, its coordinates and the page, or a
    message if the name is not found.
    """
    lines_with_coords = extract_text_and_coordinates_by_line(pdf_path)
    name_coordinates = []

    for line in lines_with_coords:
        if name in line["text"]:
            start_index = line["text"].find(name)
            end_index = start_index + len(name)
            coords = []
            current_pos = 0

            for bbox, word in line["coords"]:
                word_start = current_pos
                word_end = current_pos + len(word.strip())

                # Check if the word overlaps with the name span
                if word_start >= start_index and word_end <= end_index:
                    coords.append(bbox)

                current_pos += len(
                    word.strip()) + 1  # +1 for the space that was added in line_text

            if coords:
                name_coordinates.append({
                    "name": name,
                    "coordinates": coords,
                    "page": line["page"]
                })
                break

    if name_coordinates:
        return name_coordinates[0]  # Return the first match
    else:
        return {"message": "name not found"}


def fuzzy_match(name1, name2, threshold=90):
    """
    Performs a fuzzy match between two names with a specified threshold.

    Parameters:
    name1 (str): The first name to compare.
    name2 (str): The second name to compare.
    threshold (int): The similarity threshold (default is 90).

    Returns:
    bool: True if the similarity is above the threshold, False otherwise.
    float: The similarity score between the two names.
    """
    similarity = fuzz.ratio(name1, name2)
    return similarity > threshold, similarity


def process_pdf(pdf_path, name_list):
    """
    Processes the PDF to find names, their coordinates, and fuzzy match scores.

    Parameters:
    pdf_path (str): The path to the PDF file.
    name_list (list): A list of names to compare.

    Returns:
    dict: A dictionary containing the names found, their coordinates, and fuzzy
    match scores.
    """
    # Extract text and find names
    lines_with_coords = extract_text_and_coordinates_by_line(pdf_path)
    full_text = " ".join([line["text"] for line in lines_with_coords])
    extracted_names = extract_names(full_text)

    result = {}

    for name in extracted_names:
        # Find coordinates
        coordinates = find_name_coordinates(pdf_path, name)

        # Calculate fuzzy match scores
        matches = {}
        for input_name in name_list:
            is_match, score = fuzzy_match(name, input_name)
            matches[input_name] = {"match": is_match, "score": score}

        result[name] = {
            "coordinates": coordinates,
            "fuzzy_matches": matches
        }

    return result

