import csv
import os
import base64
from openai import OpenAI

# Define the path to the CSV file and the image directory
csv_file_path = 'cartoonData.csv'
image_directory = 'Face_extraction'
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

# Function to prompt GPT-4o with the image and text
def get_emotion(image_path, corresponding_text):
    client = OpenAI(api_key=api_key)

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that identifies emotions displayed by characters in images and text."},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    f"Here is some text and an image. They are taken from a cartoon.\n"
                    f"Your task is to take the image and text information, and label it with a maximum of two of the following seven emotions: "
                    f"Happiness, Anger, Sadness, Fear, Disgust, Surprise, or Contempt.\n"
                    f"Answer with only the emotion or emotions you identify, with a maximum of two emotions.\n\n"
                    f"Text: {corresponding_text}"
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }}
            ]}
        ],
        temperature=0.0,
    )

    # Get the response and convert it to a list
    response_text = response.choices[0].message.content.strip()
    emotions_list = [emotion.strip() for emotion in response_text.split(',')]
    
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

    with open("results.txt", "w") as results_file:
        for row in rows:
            image_name = row['Image Name']
            corresponding_text = row['Corresponding Text']
            annotation = row['Annotation']
            image_path = get_image_path(image_directory, image_name)
            
            identified_emotions = get_emotion(image_path, corresponding_text)
            all_emotions.append(identified_emotions)
            
            is_correct = check_correctness(identified_emotions, annotation)
            if is_correct:
                correct_count += 1
            
            results_file.write(f"Processed {image_name}.jpg - Correct: {is_correct}\n")

        total_rows = len(rows)
        correct_percentage = (correct_count / total_rows) * 100

        results_file.write(f"\nAll Detected Emotions: {all_emotions}\n")
        results_file.write(f"Total Correct Identifications: {correct_count}/{total_rows}\n")
        results_file.write(f"Percentage of Correct Identifications: {correct_percentage:.2f}%\n")

except Exception as e:
    with open("results.txt", "w") as results_file:
        results_file.write(f"Error: {e}\n")
