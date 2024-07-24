import csv
import os
import time
import torch
from PIL import Image, UnidentifiedImageError
from transformers import Blip2Processor, Blip2Model
import re

# Define the path to the CSV file and the image directory
csv_file_path = 'cartoonData.csv'
image_directory = 'Face_extraction'
model_name = "Salesforce/blip2-opt-2.7b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the BLIP-2 model and processor
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2Model.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# Function to read all rows from the CSV file
def get_all_rows(csv_file_path):
    with open(csv_file_path, mode='r', encoding='latin1') as file:
        csv_reader = csv.DictReader(file)
        rows = [row for row in csv_reader]
    return rows

# Function to find the corresponding image file
def get_image_path(image_directory, image_name):
    image_path = os.path.join(image_directory, f"{image_name}")
    if os.path.exists(image_path):
        return image_path
    else:
        raise FileNotFoundError(f"Image '{image_path}' not found.")

# Function to encode the image as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to resize the image and convert to RGB
def resize_image(image_path, output_path, size=(128, 128)):
    with Image.open(image_path) as img:
        img = img.resize(size).convert("RGB")
        img.save(output_path, format="JPEG")

# Function to prompt BLIP-2 with the image and text
def get_emotion(image_path, corresponding_text, same_character):
    try:
        resized_image_path = "/tmp/resized_image.jpg"
        resize_image(image_path, resized_image_path)
        image = Image.open(resized_image_path).convert("RGB")

        # Prepare the prompt
        prompt = (
            f"Here is some text and an image. They are taken from a cartoon.\n"
            f"Your task is to identify the emotion(s) being displayed by them. "
            f"Choose one or two emotions from the following seven: Happiness, Anger, Sadness, Fear, Disgust, Surprise, or Contempt.\n"
            f"Answer in the format: [Emotion] or [Emotion, Emotion].\n\n"
            f"Text: {corresponding_text[:250]}\n\n"
            f"Said by same character?: {same_character}"
        )

        # Process the input
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        # Generate the response
        output = model.generate(**inputs)
        response_text = processor.decode(output[0], skip_special_tokens=True).strip()

        # Extract emotions from the response
        emotions = re.findall(r'\b(Happiness|Anger|Sadness|Fear|Disgust|Surprise|Contempt)\b', response_text, re.IGNORECASE)
        unique_emotions = []
        for emotion in emotions:
            if emotion.lower() not in [e.lower() for e in unique_emotions]:
                unique_emotions.append(emotion)
            if len(unique_emotions) == 2:
                break

        return unique_emotions
    except Exception as e:
        return []  # Return a blank list in case of error

# Function to check correctness of identified emotions
def check_correctness(identified_emotions, annotation):
    annotated_emotions = [emotion.strip() for emotion in annotation.split(',')]
    return any(emotion in identified_emotions for emotion in annotated_emotions)

# Main script execution
try:
    rows = get_all_rows(csv_file_path)
    all_emotions = []
    correct_count = 0

    with open("blip2Results.txt", "w") as results_file:
        # Print the column headers to debug
        if rows:
            print("Column headers:", rows[0].keys())
            results_file.write(f"Column headers: {list(rows[0].keys())}\n")

        for row in rows:
            image_name = row.get('Image Name', '').strip()
            corresponding_text = row.get('Corresponding Text', '').strip()
            same_character = row.get('Said by same character?', '').strip()
            annotation = row.get('Annotation', '').strip()
            
            if not image_name or not corresponding_text or not annotation:
                results_file.write(f"Skipping row due to missing data: {row}\n")
                all_emotions.append([])  # Add empty list as a placeholder
                continue
            
            try:
                image_path = get_image_path(image_directory, image_name)
                identified_emotions = get_emotion(image_path, corresponding_text, same_character)
            except (FileNotFoundError, UnidentifiedImageError) as e:
                results_file.write(f"Skipping row due to error: {e}\n")
                all_emotions.append([])  # Add empty list as a placeholder
                continue
            
            all_emotions.append(identified_emotions)
            
            is_correct = check_correctness(identified_emotions, annotation)
            if is_correct:
                correct_count += 1
            
            results_file.write(f"Processed {image_name} - Correct: {is_correct}\n")
            time.sleep(5)  # Adding a 5-second delay between calls

        total_rows = len(rows)
        correct_percentage = (correct_count / total_rows) * 100

        results_file.write(f"\nAll Detected Emotions: {all_emotions}\n")
        results_file.write(f"Total Correct Identifications: {correct_count}/{total_rows}\n")
        results_file.write(f"Percentage of Correct Identifications: {correct_percentage:.2f}%\n")

except Exception as e:
    with open("blip2Results.txt", "w") as results_file:
        results_file.write(f"Error: {e}\n")
