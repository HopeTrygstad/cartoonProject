import csv
import os
import requests
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

# Set device
device = "cuda:0"

# Load the pretrained LLaVa-Next model and processor
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

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

# Function to prompt LLaVa-Next with the image and text
def get_emotion(image_path, corresponding_text, same_character):
    try:
        print(f"Opening image: {image_path}")
        image = Image.open(image_path)
        prompt = f"[INST] <image>\nText: {corresponding_text}\nSaid by same character?: {same_character}\n Given these three pieces of information, what emotion is this character displaying? [/INST]"
        print(f"Prompt: {prompt}")

        inputs = processor(prompt, image, return_tensors="pt").to(device)
        print(f"Inputs: {inputs}")

        outputs = model.generate(**inputs, max_length=50)
        print(f"Outputs: {outputs}")

        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")

        emotions_list = [emotion.strip() for emotion in response.split(',')]
        return emotions_list
    except Exception as e:
        print(f"Error during emotion generation: {e}")
        return ["Error"]

# Test processing of the first row
try:
    print("Reading rows from CSV file...")
    rows = get_all_rows(csv_file_path)
    print(f"Total rows read: {len(rows)}")

    if rows:
        first_row = rows[0]
        print(f"First row: {first_row}")

        image_name = first_row.get('Image Name', '').strip()
        corresponding_text = first_row.get('Corresponding Text', '').strip()
        same_character = first_row.get('Said by same character?', '').strip()
        annotation = first_row.get('Annotation', '').strip()

        if not image_name or not corresponding_text or not annotation:
            print(f"Skipping row due to missing data: {first_row}")
        else:
            image_path = get_image_path(image_directory, image_name)
            identified_emotions = get_emotion(image_path, corresponding_text, same_character)
            print(f"Identified Emotions: {identified_emotions}")

except Exception as e:
    print(f"Error: {e}")
