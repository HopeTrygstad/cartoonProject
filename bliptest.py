from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import csv
import os
import re


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT

# Define the path to the CSV file and the image directory
csv_file_path = 'cartoonData.csv'
image_directory = 'Face_extraction'

# Function to read all rows from the CSV file
def get_all_rows(csv_file_path):
    with open(csv_file_path, mode='r', encoding='latin1') as file:
        csv_reader = csv.DictReader(file)
        rows = [row for row in csv_reader]
    return rows

# Function to find the corresponding image file
def get_image_path(image_directory, image_name):
    image_path = os.path.join(image_directory, image_name)
    if os.path.exists(image_path):
        return image_path
    else:
        raise FileNotFoundError(f"Image '{image_path}' not found.")

def get_emotion(image_path, corresponding_text, same_character):
    try:
        image = Image.open(image_path).convert("RGB")
        prompt = (
            "Here is some text and an image. They are taken from a cartoon.\n"
            "The image is a frame from the cartoon with a character's face on it.\n"
            "The text is the piece of dialogue that was being said during the cartoon at the time of the frame. \n"
            "'Said by same character?' indicates whether or not the text was said by the character in the image.\n"
            "Question: out of the following seven emotions, what 1-2 emotions are being displayed? \n"
            "Happiness, Anger, Sadness, Fear, Disgust, Surprise, or Contempt.\n"
            "Answer(1-2 words): \n\n"
            f"Text: \"{corresponding_text}\"\n"
            f"Said by same character?: {same_character}\n"
        )

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
        generated_ids = model.generate(**inputs)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(f"Response for image {image_path}: {response}")  # Debugging statement

    except Exception as e:
        print(f"Error processing {image_path}: {e}")  # Debugging statement
        return [f"Error: {e}"]


try:
    rows = get_all_rows(csv_file_path)
    all_emotions = []
    correct_count = 0

    with open("blip2Results.txt", "w") as results_file:
        # Print the column headers to debug
        if rows:
            results_file.write(f"Column headers: {list(rows[0].keys())}\n")
            results_file.flush()

        for idx, row in enumerate(rows):
            print(f"Processing row {idx+1}/{len(rows)}")
            image_name = row.get('Image Name', '').strip()
            corresponding_text = row.get('Corresponding Text', '').strip()
            same_character = row.get('Said by same character?', '').strip()
            annotation = row.get('Annotation', '').strip()

            if not image_name or not corresponding_text or not annotation:
                results_file.write(f"Skipping row due to missing data: {row}\n")
                results_file.flush()
                continue

            image_path = get_image_path(image_directory, image_name)
            get_emotion(image_path, corresponding_text, same_character)
