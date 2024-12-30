import os
import json
from PIL import Image
import shutil
from tqdm import tqdm

# vraiables
output_path_dup = 'final_dataset/duplicate.json'
output_path = 'final_dataset/plasticGarbageDataset'
if os.path.exists(output_path):
    # Folder exists, delete all contents inside it
    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename) 
        # Check if it's a file or directory
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)  # Delete the folder and its contents
        else:
            os.remove(file_path)
# Create the folder if it doesn't exist
else:
    os.makedirs(output_path)

# Check if the file exists
if os.path.exists(output_path_dup):
    # Load the JSON data
    with open(output_path_dup, 'r') as file:
        data = json.load(file)
        ban_images = [obj['duplicate'] for obj in data]
else:
    ban_images = []

# True if multiclass
paths = [
    (r'Dataset-20241229T161246Z-001\Dataset/test',False),
    (r'Dataset-20241229T161246Z-001\Dataset/train',False),
    (r'Dataset-20241229T161246Z-001\Dataset/valid',False),
    (r'archive/underwater_plastics/test',True),
    (r'archive/underwater_plastics/train',True),
    (r'archive/underwater_plastics/valid',True),
    
]

labels = {}
total_data = {}
for path, multiclass in paths:
    current_label = path.split('/')[-1]
    if current_label not in total_data:
        total_data[current_label] = 0

    image_path = os.path.join(path, 'images')
    imagenames = os.listdir(image_path)
    label_path = os.path.join(path, 'labels')
    labelnames = os.listdir(label_path)
    for image_name in tqdm(imagenames, desc=f'Process {path}'):
        for label_name in labelnames:
            if image_name.split('.jpg')[0] == label_name.split('.txt')[0]:
                filepath = os.path.join(label_path, label_name)
                f = open(filepath, 'r')
                if multiclass:
                    lines = f.readlines()
                    labels = []
                    for line in lines:
                        label = line.split()
                        label = [i for i in label]
                        label[0] = '0'
                        labels.append(' '.join(label))
                    lines = '\n'.join(labels)
                else:
                    lines = f.read()
                f.close()

                # save 
                current_output_path = os.path.join(output_path, current_label)
                if not os.path.exists(current_output_path):
                    # Create the folder if it doesn't exist
                    os.makedirs(current_output_path)
                    os.makedirs(current_output_path + '/labels')
                    os.makedirs(current_output_path + '/images')
        
                # Open the image
                with Image.open(os.path.join(image_path, image_name)) as img:
                    # Save the image as JPG
                    img.save(os.path.join(current_output_path + '/images', image_name), 'JPEG')

                # save label
                labelfile = open(os.path.join(current_output_path + '/labels', label_name), 'w')
                labelfile.write(lines)
                labelfile.close()

                total_data[current_label] += 1
                break

total_data['total'] = sum(total_data.values())
print(total_data)