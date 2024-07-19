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
        prompt = (
            "[INST] <image>\n"
            "Here is some text and an image. They are taken from a cartoon.\n"
            "Your task is to take the image and text information, and label it with a maximum of two of the following seven emotions: "
            "Happiness, Anger, Sadness, Fear, Disgust, Surprise, or Contempt.\n"
            "Answer with only the emotion or emotions you identify, with a maximum of two emotions.\n\n"
            f"Text: {corresponding_text}\n"
            f"Said by same character?: {same_character} [/INST]"
        )
        print(f"Prompt: {prompt}")

        inputs = processor(prompt, image, return_tensors="pt").to(device)

        outputs = model.generate(**inputs, max_new_tokens=100)

        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")

        emotions_list = [emotion.strip() for emotion in response.split(',')]
        return emotions_list
    except Exception as e:
        print(f"Error during emotion generation: {e}")
        return ["Error"]

# Function to check correctness of identified emotions
def check_correctness(identified_emotions, annotation):
    annotated_emotions = [emotion.strip() for emotion in annotation.split(',')]
    return any(emotion in identified_emotions for emotion in annotated_emotions)

# Main script execution
try:
    rows = get_all_rows(csv_file_path)
    all_emotions = []
    correct_count = 0

    with open("LlaVaResults.txt", "w") as results_file:
        # Print the column headers to debug
        if rows:
            print("Column headers:", rows[0].keys())
            results_file.write(f"Column headers: {list(rows[0].keys())}\n")

        for i, row in enumerate(rows, 1):
            image_name = row.get('Image Name', '').strip()
            corresponding_text = row.get('Corresponding Text', '').strip()
            same_character = row.get('Said by same character?', '').strip()
            annotation = row.get('Annotation', '').strip()
            
            if not image_name or not corresponding_text or not annotation:
                results_file.write(f"Skipping row due to missing data: {row}\n")
                continue
            
            image_path = get_image_path(image_directory, image_name)
            identified_emotions = get_emotion(image_path, corresponding_text, same_character)
            all_emotions.append(identified_emotions)
            
            is_correct = check_correctness(identified_emotions, annotation)
            if is_correct:
                correct_count += 1
            
            results_file.write(f"Processed {image_name} - Correct: {is_correct}\n")

        total_rows = len(rows)
        correct_percentage = (correct_count / total_rows) * 100

        results_file.write(f"\nAll Detected Emotions: {all_emotions}\n")
        results_file.write(f"Total Correct Identifications: {correct_count}/{total_rows}\n")
        results_file.write(f"Percentage of Correct Identifications: {correct_percentage:.2f}%\n")

except Exception as e:
    with open("LlaVaResults.txt", "w") as results_file:
        results_file.write(f"Error: {e}\n")
