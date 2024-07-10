import csv
import os
from openai import OpenAI

# Define the path to the CSV file and the image directory
csv_file_path = 'cartoonData.csv'
image_directory = 'Face_extraction'
api_key = os.getenv('OPENAI_API_KEY')

# Function to read the first row from the CSV file
def get_first_row(csv_file_path):
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            return row['Image Name'], row['Corresponding Text']

# Function to find the corresponding image file
def get_image_path(image_directory, image_name):
    image_path = os.path.join(image_directory, f"{image_name}")
    if os.path.exists(image_path):
        return image_path
    else:
        raise FileNotFoundError(f"Image '{image_path}' not found.")

# Function to prompt GPT-4o with the text
def get_emotion(corresponding_text):
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant skilled in emotion detection from text and images."},
            {"role": "user", "content": f"""Here is some text and/or image input. 
             Your task is to take the data, and label it with a maximum of two of the following seven emotions: 
             Happiness, Anger, Sadness, Fear, Disgust, Surprise, or Contempt.
             Answer with only the emotion or emotions you identify, with a maximum of two emotions. 
              Determine the emotion:\n\nText: {corresponding_text}"""}
        ]
    )

    return completion.choices[0].message.content

# Main script execution
try:
    image_name, corresponding_text = get_first_row(csv_file_path)
    image_path = get_image_path(image_directory, image_name)  # Verify the image exists
    emotion = get_emotion(corresponding_text)
    print(f"Image: {image_path}")
    print(f"Detected Emotion: {emotion}")
except Exception as e:
    print(f"Error: {e}")
