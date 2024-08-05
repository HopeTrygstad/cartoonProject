import csv
import os
import re
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

# Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the pretrained LLaVa-Next model and processor
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Define the path to the CSV file and the image directory
csv_file_path = 'squidward_dataset.csv'
image_directory = os.path.expanduser('~/Downloads/cartoonProject/squidward_faces')

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
        image = Image.open(image_path)
        prompt = (
            "[INST] <image>\n"
            "Here is some text and an image. They are taken from a cartoon.\n"
            "The image is a frame from the cartoon with a character's face on it.\n"
            "The text is the piece of dialogue that was being said during the cartoon at the time of the frame. \n"
            "'Said by same character?' indicates whether or not the text was said by the character in the image.\n"
            "Your task is to take the image and text information, and label it with a maximum of two of the following seven emotions: "
            "Happiness, Anger, Sadness, Fear, Disgust, Surprise, or Contempt.\n"
            "Answer with only the emotion or emotions you identify, with a maximum of two emotions.\n\n"
            f"Text: \"{corresponding_text}\"\n"
            f"Said by same character?: {same_character}\n"
            "[/INST]"
        )

        inputs = processor(prompt, image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = processor.decode(outputs[0], skip_special_tokens=True)

        # Extract emotions from the response
        emotions_list = re.findall(r'\b(Happiness|Anger|Sadness|Fear|Disgust|Surprise|Contempt)\b', response)

        return emotions_list
    except Exception as e:
        return [f"Error: {e}"]

# Function to check correctness of identified emotions
def check_correctness(identified_emotions, annotation):
    annotated_emotions = [emotion.strip() for emotion in annotation.split(',')]
    return any(emotion in identified_emotions for emotion in annotated_emotions)

# Main script execution
try:
    rows = get_all_rows(csv_file_path)
    all_emotions = []
    correct_count = 0

    with open("LlavaResults.txt", "w") as results_file:
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
    with open("LlavaResults.txt", "w") as results_file:
        results_file.write(f"Error: {e}\n")
        results_file.flush()
