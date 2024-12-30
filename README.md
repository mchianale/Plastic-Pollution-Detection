# Plastic Pollution Detection in Marine Environments

![example](https://github.com/mchianale/Plastic-Pollution-Detection/blob/main/doc/val_batch2_pred.jpg)

---

## Introduction

Plastic pollution has become a pressing environmental issue, significantly impacting marine ecosystems worldwide. Identifying, classifying, and detecting plastic waste, both on the water's surface and underwater, is critical for understanding pollution patterns and implementing effective mitigation strategies. Leveraging advanced machine learning models, this project aims to detect, classify, and analyze plastic waste in aquatic environments, combining datasets from diverse sources and cutting-edge algorithms like YOLO and CLIP.

Our work encompasses three primary objectives:
1. **Dataset Preparation**: Curating and refining datasets with diverse marine imagery for effective model training and evaluation.
2. **Object Detection**: Training and fine-tuning YOLO for detecting plastic waste in underwater environments.
3. **Image Classification**: Utilizing the CLIP model to classify images into surface or underwater categories, enhancing analysis precision.

By employing these techniques, this project addresses key challenges in marine conservation, offering insights into plastic pollution patterns and aiding in its mitigation.

--- 

## Sommaire

1. [Datasets](#datasets)  
   1.1 [DeepTrash Dataset](#1-deeptrash-dataset)  
   1.2 [Underwater Plastic Pollution Detection Dataset](#2-underwater-plastic-pollution-detection-dataset)  
   1.3 [Final Dataset](#final-dataset)  
   1.4 [Managing Similar Images](#manage-similar-images)  

2. [Plastic Detection in Ocean Using YOLO](#plastic-detection-in-ocean-using-yolo)  
   2.1 [Training](#train)  
   2.2 [Evaluation](#evaluation)  

3. [CLIP Model for Image Classification: Surface or Underwater](#clip-model-for-image-classification-surface-or-underwater)  
   3.1 [Goal](#goal-why-classify-images-into-surface-or-underwater)  
   3.2 [Implementation Overview](#implementation-overview)  
   3.3 [Evaluation Process](#evaluation-process)  
   3.4 [Results and Insights](#results-and-insights)  

4. [Source](#source)

---

## Datasets
For each dataset, we created a quick `analysis.ipynb` to examine the annotated instance distribution, their spatial distribution, and other related aspects.

### 1. DeepTrash Dataset
- **Source**: [DeepPlastic Project](https://github.com/gautamtata/DeepPlastic?tab=readme-ov-file#object-detection-model)
- **Description**:
  - Field images from Lake Tahoe, San Francisco Bay, and Bodega Bay (California).
  - <20% of images are sourced from Google Images.
  - Deep sea images are taken from the JAMSTEK JEDI dataset ([JAMSTEK JEDI Dataset](http://www.godac.jamstec.go.jp/)).
- **Structure**:
  - 1,900 training images.
  - 637 test images.
  - 637 validation images.
  - Split: 60% training, 20% validation, 20% testing.
- **Download**: [Google Drive Link](https://drive.google.com/drive/folders/1fsS_u2QpbRGynYkP6-D6cfvq8r0hpjXI)

### 2. Underwater Plastic Pollution Detection Dataset
- **Source**: [Ocean Waste Dataset on Roboflow](https://universe.roboflow.com/object-detect-dmjpt/ocean_waste/dataset/1)
- **Description**:
  - Depicts underwater scenes polluted with garbage and debris.
  - Preprocessed using the Dark Prior Channel method to enhance contrast and facilitate detection.
- **Structure**:
  - **Train Directory**: 3,628 images with labels.
  - **Validation Directory**: 1,001 images with labels.
  - **Test Directory**: 501 images with labels.
- **Classes**:
  - Number of Classes: 15.
  - Class Names: 'Mask', 'Can', 'Cellphone', 'Electronics', 'Gbottle', 'Glove', 'Metal', 'Misc', 'Net', 'Pbag', 'Pbottle', 'Plastic', 'Rod', 'Sunglasses', 'Tire'.

### Final Dataset
- The final dataset merges and simplifies the previous datasets into a single class: `Trash`.
- **Data Structure**:
  - **Train**: 5,504 images.
  - **Validation**: 1,625 images.
  - **Test**: 1,114 images.
  - **Total**: 8,243 images.
  - **Total Trash Instances**: 16,121 across the entire dataset.

**Data YAML Structure**:
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['Trash']
```

### Manage Similar Images

In this section, we will demonstrate how to detect and manage duplicate images in the dataset. This can be useful for cleaning up redundant data, improving training efficiency, and ensuring that your dataset contains unique images for model training. The method leverages cosine similarity to compare image embeddings, identifying duplicates based on a similarity threshold.

**Goal**
The goal of this script is to identify similar or duplicate images in the dataset by calculating the cosine similarity between the embeddings of the images. If the similarity score between two images exceeds a specified threshold, they are considered duplicates.

**Key Steps**
- Image Preprocessing: Images are loaded and preprocessed using `OpenCLIP` and transformed into embeddings.
- Cosine Similarity: The embeddings are compared pairwise using cosine similarity to determine how similar two images are.
- Thresholding: If the similarity score between two images is greater than or equal to a defined threshold (default is 0.95), they are flagged as duplicates.
- Output: The script outputs a JSON file containing a list of duplicate images, with their respective similarity scores.

**Parameters to Change**
- `images_path`: This list defines the directories containing the images you want to check for duplicates.
- `output_path`: Specify the path where the resulting JSON file containing the duplicates will be.
- `threshold`: This parameter defines the cosine similarity threshold above which two images will be considered duplicates. The default value is 0.95, meaning that two images must be at least 95% similar to be flagged as duplicates. You can adjust this value based on your requirements.

**Run**
```bash
python final_dataset\removeDuplicate.py
```

**Check Output**: After the script runs, the duplicate.json file will contain a list of duplicate images and their similarity scores, formatted like this:
```json
[
    {
        "duplicate": "path_to_image_1.jpg",
        "equal_to": "path_to_image_2.jpg",
        "similarity": 98.25
    },
]
```

**Improvement**
While the image duplication detection method presented here is effective, we ultimately decided not to use it on our dataset for the following reasons:

- Optimization Needs: The code, as written, is computationally expensive. It processes each pair of images in the dataset, which results in an O(n²) time complexity. As the dataset grows, this becomes increasingly costly in terms of both time and memory usage.

---

## Plastic Detection in Ocean using YOLO

In this section, we describe the model used for detecting plastic pollution in ocean environments. The model is based on [Ultralytics YOLOv11](https://docs.ultralytics.com/fr/models/yolo11/), a popular and efficient object detection algorithm known for its high speed and accuracy. We specifically use a pre-trained YOLO model (`yolo11n.pt`) for detecting various types of plastic waste in underwater scenes.

Look notebook `yolo_run.ipynb`. 

### Train 
**Parameters :**
- `epochs`: 23
- `batch`: 16
- `lr0`: 0.01
- `dropout`: 0.15
- `imgsz`: 64

**Note :** We trained the model using a Google Colab notebook with the free T4 GPU, leveraging the cloud-based resources to accelerate the training process.

### Evaluation

For evaluating the performance of our YOLO model in detecting plastic waste, we utilized the **Intersection over Union (IoU)** metric to compare predicted bounding boxes with ground truth labels. IoU is a commonly used metric in object detection tasks, which measures the overlap between the predicted and actual bounding boxes.

#### Evaluation Methodology

1. **IoU Threshold**: We fixed the IoU threshold at **70%**. This means:
   - Predictions with an IoU ≥ 70% with the ground truth are considered accurate detections.
   - Predictions with an IoU < 70% are classified as **bad localization**.

2. **Performance Metrics**:
   - **Missed Predictions**: Count of ground truth objects that the model failed to detect.
   - **Background Predictions**: Predictions made by the model for objects that don't exist in the image (false positives).
   - **Duplicate Predictions**: Multiple predictions for the same object.
   - **Bad Localization**: Predictions that fail to meet the IoU threshold.

3. **Results on the Test Dataset**:
   - The following table summarizes the evaluation results for our fine-tuned YOLO model on the test dataset:

| Missed | Background | Duplicates | Bad Localization | Mean IoU | Total Detections |
|--------|------------|------------|-------------------|----------|------------------|
| 226    | 332        | 911        | 456               | 0.692268 | 1699            |

#### Insights:
- The model achieved an average **Mean IoU** of **0.692**, which is slightly below the desired threshold of 70%. This indicates that while the model performed reasonably well in detecting objects, improvements can be made in localization accuracy.
- **Missed Predictions** and **Background Predictions** highlight the need for more balanced training data or additional fine-tuning.
- **Duplicate Predictions** suggest that the model sometimes detects the same object multiple times, which could be improved by implementing post-processing techniques such as Non-Maximum Suppression (NMS).

#### Future Improvements:
To address these issues, we could:
- Increase the diversity and quality of the training dataset to reduce missed and background predictions.
- Adjust the IoU threshold and explore alternative metrics such as F1-score or precision-recall curves.
- Refine post-processing techniques to mitigate duplicate predictions and improve localization accuracy.

---

## CLIP Model for Image Classification: Surface or Underwater

In this section, we utilized OpenAI's CLIP model (`clip-vit-large-patch14`) to classify images as representing either the **surface of the water** or **under the water**. This approach leverages CLIP's ability to compute similarity between textual and visual representations, making it a powerful tool for semantic image classification.

### Goal: Why Classify Images into Surface or Underwater?
The classification of images into "surface of the water" or "under the water" serves several important purposes:

1. **Improved Object Detection**: Images from underwater scenes often exhibit different visual characteristics, such as altered lighting and contrast, compared to surface images. Classifying these images enables better preprocessing and more effective fine-tuning of object detection models.

2. **Environmental Analysis**: Differentiating between surface and underwater images allows researchers to focus on specific environmental issues, such as tracking floating plastic debris on the water's surface or examining marine life and pollution below the water.

3. **Data Organization**: Sorting images into distinct categories ensures a well-organized dataset, which simplifies further analysis and machine learning workflows.

### Implementation Overview
The CLIP model was used to assign each image to one of two predefined categories: "surface of the water" or "under the water." For each image:
- Textual descriptions were compared to the image using CLIP's similarity scoring mechanism.
- The category with the highest similarity score was selected as the label for the image.
- Predictions were stored for evaluation.

### Evaluation Process

To assess the model's performance, a subset of the test images was manually annotated. This allowed us to compare the model's predictions against ground truth labels, providing insight into its classification accuracy.

### Results and Insights

#### Observations
- The CLIP model performed well on clear and distinct images, where the characteristics of "surface" or "underwater" scenes were visually prominent.
- Ambiguous or low-quality images sometimes led to incorrect classifications due to unclear features or lighting.
- Accuracy on 100 annoted images : **81%**

#### Next Steps
1. **Model Fine-Tuning**: Fine-tuning CLIP with a small annotated dataset could significantly improve classification accuracy, especially for ambiguous cases.
2. **Preprocessing Enhancements**: For underwater images, applying preprocessing methods such as contrast adjustment or dehazing might improve classification performance.
3. **Extending Applications**: This classification approach could be integrated into broader pipelines for marine monitoring and environmental conservation.

By leveraging CLIP's capability to understand image-text relationships, this workflow demonstrated an efficient and versatile method for classifying marine images into surface and underwater categories.

---

## Source

- [Yolov11](https://docs.ultralytics.com/fr/models/yolo11/)
- [DeepPlastic Project](https://github.com/gautamtata/DeepPlastic?tab=readme-ov-file#object-detection-model)
- [Ocean Waste Dataset](https://universe.roboflow.com/object-detect-dmjpt/ocean_waste/dataset/1)
- [CLIP model](https://github.com/openai/CLIP/blob/main/model-card.md)
