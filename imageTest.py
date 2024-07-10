import csv
import os
import base64
from openai import OpenAI

# Define the path to the CSV file and the image directory
csv_file_path = 'cartoonData.csv'
image_directory = 'Face_extraction'
api_key = '***REMOVED***'
model = "gpt-4o"

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

# Function to encode the image as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to prompt GPT-4o with the image and text
def get_emotion(image_path):
    client = OpenAI(api_key=api_key)

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that identifies emotions displayed by characters in images."},
            {"role": "user", "content": [
                {"type": "text", "text": """Here is some text and/or image input. 
             Your task is to take the data, and label it with a maximum of two of the following seven emotions: 
             Happiness, Anger, Sadness, Fear, Disgust, Surprise, or Contempt.
             Answer with only the emotion or emotions you identify, with a maximum of two emotions."""},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }}
            ]}
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content

# Main script execution
try:
    image_name, _ = get_first_row(csv_file_path)
    image_path = get_image_path(image_directory, image_name)
    
    emotion = get_emotion(image_path)
    print(f"Image: {image_path}")
    print(f"Detected Emotion: {emotion}")
except Exception as e:
    print(f"Error: {e}")
