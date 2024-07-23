import csv
import os
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the pretrained BLIP-2 model and processor
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

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

# Function to prompt BLIP-2 with the image and text
def get_emotion(image_path, corresponding_text, same_character):
    try:
        image = Image.open(image_path)
        prompt = (
            f"Here is an image and some text from a cartoon. The image is a frame from the cartoon with a character's face. "
            f"The text is a piece of dialogue said during this frame. 'Said by same character?' indicates whether the text was said by the character in the image. "
            f"Your task is to take the image and text information, and label it with the top two emotions displayed by the character. "
            f"Choose from: Happiness, Anger, Sadness, Fear, Disgust, Surprise, or Contempt. "
            f"Text: \"{corresponding_text}\" "
            f"Said by same character?: {same_character}. "
            f"Answer only with the two emotions, separated by a comma."
        )

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(f"Raw response: {generated_text}")

        detected_emotions = [emotion.strip().capitalize() for emotion in generated_text.split(',') if emotion.strip()]
        
        print(f"Detected emotions: {detected_emotions}")

        return detected_emotions

    except Exception as e:
        print(f"Error during emotion generation: {e}")
        return ["Error"]

# Function to check correctness of identified emotions
def check_correctness(identified_emotions, annotation):
    annotated_emotions = [emotion.strip().capitalize() for emotion in annotation.split(',')]
    return any(emotion in identified_emotions for emotion in annotated_emotions)

# Main script execution
try:
    rows = get_all_rows(csv_file_path)
    all_emotions = []
    correct_count = 0

    with open("Llava2GuessesResults.txt", "w") as results_file:
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
    with open("Llava2GuessesResults.txt", "w") as results_file:
        results_file.write(f"Error: {e}\n")
        results_file.flush()
