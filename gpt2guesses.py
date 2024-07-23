import csv
import os
import base64
import time
import openai
from PIL import Image, UnidentifiedImageError

# Define the path to the CSV file and the image directory
csv_file_path = '/Users/hopetrygstad/Downloads/cartoonProject/cartoonData.csv'
image_directory = '/Users/hopetrygstad/Downloads/cartoonProject/Face_extraction'
api_key = os.getenv('OPENAI_API_KEY')
model = "gpt-4o"

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

# Function to prompt GPT-4o with the image and text
def get_emotion(image_path, corresponding_text, same_character):
    openai.api_key = api_key

    resized_image_path = "/tmp/resized_image.jpg"
    resize_image(image_path, resized_image_path)

    base64_image = encode_image(resized_image_path)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that identifies emotions displayed by characters in images and text."},
            {"role": "user", "content": (
                f"Here is some text and an image. They are taken from a cartoon.\n"
                f"Your task is to take the image and text information, and label your best guess from the following seven emotions: "
                f"Happiness, Anger, Sadness, Fear, Disgust, Surprise, or Contempt.\n"
                f"After you provide your best guess, think about what your second guess would be, and provide your second best guess for what emotion is being displayed."
                f"Answer with exactly two emotions, in the format [Emotion, Emotion].\n\n"
                f"Text: {corresponding_text[:250]}\n\n"
                f"Said by same character?: {same_character}"
            )},
            {"role": "user", "content": f"data:image/jpeg;base64,{base64_image}"}
        ],
        temperature=0.0,
    )

    # Get the response and convert it to a list
    response_text = response['choices'][0]['message']['content'].strip()
    emotions_list = [emotion.strip() for emotion in response_text.strip('[]').split(',')]
    
    return emotions_list

# Function to check correctness of identified emotions
def check_correctness(identified_emotions, annotation):
    annotated_emotions = [emotion.strip() for emotion in annotation.split(',')]
    return any(emotion in identified_emotions for emotion in annotated_emotions)

# Main script execution
try:
    rows = get_all_rows(csv_file_path)
    all_emotions = []
    correct_count = 0

    with open("gpt2guessesResults.txt", "w") as results_file:
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
    with open("gpt2guessesResults.txt", "w") as results_file:
        results_file.write(f"Error: {e}\n")
