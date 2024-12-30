import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
import json 
import os
from tqdm import tqdm


# variables 
images_path = [
    r'archive\underwater_plastics\test\images',
    r'archive\underwater_plastics\train\images',
    r'archive\underwater_plastics\valid\images',
    r'Dataset-20241229T161246Z-001\Dataset\test\images',
    r'Dataset-20241229T161246Z-001\Dataset\train\images',
    r'Dataset-20241229T161246Z-001\Dataset\valid\images'
]
output_path = 'final_dataset/duplicate.json'
threshold = 0.95

# image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)


def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def generateScore(img1, image2):
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = float(cos_scores[0][0])
    return score



# start get all file path
file_names = []
visit = []
for image_path in images_path:
    # List all files in the folder
    current_file_names = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    file_names += current_file_names
    visit += [0 for _ in range(len(current_file_names))]

# similiraty 
duplicates = []
for i in tqdm(range(len(file_names)), desc='compute similiraties'):
    if visit[i] == 1:
        continue
    visit[i] = 1
    # load image
    test_img = cv2.imread(file_names[i], cv2.IMREAD_UNCHANGED)
    img = imageEncoder(test_img)
    for j in tqdm(range(len(file_names)), desc=f'current iteration {i}'):
        if visit[j] == 1:
            continue
        score = generateScore(img, file_names[j])
        if score >= threshold:
            duplicates.append({'duplicate' : file_names[j], 'equal_to' : file_names[i], 'similiraty' : score*100})
            visit[j] = 1

# save result
with open(output_path, "w") as json_file:
    json.dump(duplicates, json_file, indent=4) 
    