from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch
import csv
import os
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", quantization_config=quantization_config, device_map={"": 0}, torch_dtype=torch.float16
)

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
            "Your task is to take the image and text information, and label it with a maximum of two of the following seven emotions: "
            "Happiness, Anger, Sadness, Fear, Disgust, Surprise, or Contempt.\n"
            "Label the emotions displayed. Answer with one or two emotions.\n"
            f"Text: \"{corresponding_text}\"\n"
            f"Said by same character?: {same_character}\n"
        )

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if response:
            print(f"Response for image {image_path}: {response}")  # Debugging statement
        else:
            print(f"No response for image {image_path}")

        return response

    except Exception as e:
        print(f"Error processing {image_path}: {e}")  # Debugging statement
        return [f"Error: {e}"]

def check_correctness(identified_emotions, annotation):
    if identified_emotions is None:
        return False
    annotated_emotions = [emotion.strip() for emotion in annotation.split(',')]
    return any(emotion in identified_emotions for emotion in annotated_emotions)

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
            identified_emotions = get_emotion(image_path, corresponding_text, same_character)
            all_emotions.append(identified_emotions)

            is_correct = check_correctness(identified_emotions, annotation)
            if is_correct:
                correct_count += 1

            results_file.write(f"Processed {image_name} - Correct: {is_correct}\n")
            results_file.flush()

        total_rows = len(rows)
        correct_percentage = (correct_count / total_rows) * 100

        results_file.write(f"\nAll Detected Emotions: {all_emotions}\n")
        results_file.write(f"Total Correct Identifications: {correct_count}/{total_rows}\n")
        results_file.write(f"Percentage of Correct Identifications: {correct_percentage:.2f}%\n")
        results_file.flush()

except Exception as e:
    with open("blip2Results.txt", "w") as results_file:
        results_file.write(f"Error: {e}\n")
        results_file.flush()
