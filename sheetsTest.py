import csv

# Define the path to the CSV file
csv_file_path = 'cartoonData.csv'
# Define the subdirectory where images are stored
image_directory = 'Face_extraction/'

# Open and read the CSV file
with open(csv_file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    
    # Iterate through each row in the CSV
    for row in csv_reader:
        # Get the image name, corresponding text, and "Said by Same Character?" column
        image_name = row['Image Name']
        corresponding_text = row['Corresponding Text']
        said_by_same_character = row['Said by same character?']
        
        # Print the image name, corresponding text, and "Said by Same Character?" column
        print(f"Image: {image_directory}{image_name}, Text: {corresponding_text}, Said by Same Character?: {said_by_same_character}")
