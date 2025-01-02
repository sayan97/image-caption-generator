# Image Caption Generator
This repository contains an implementation of an Image Caption Generator web application that uses a pre-trained CNN (ResNet-152) encoder and RNN decoder architecture to generate descriptive captions for images. The CNN based ResNet-152 encoder extracts deep visual features from images and the RNN decoder translates these features into human-readable captions. The model was trained on a subset of [COCO "Common Objects in Context" dataset](https://cocodataset.org/#home), known for its rich variety of image-caption pairs of 80 object categories, and at least five textual reference captions per image.

# Demo
https://github.com/user-attachments/assets/709f3ad1-d925-4772-b577-32a5e5ed440d

# How to use this repository
- **Step1:** Clone the repository `git clone https://github.com/sayan97/image-caption-generator.git`
- **Step2:** Change directory to `image-caption-generator`
- **Step3:** Create a virtual environment `conda create -n image-caption-generator python==3.8`
- **Step3:** Activate the virtual environment `conda activate image-caption-generator`
- **Step4:** Install required libraries `pip install -r requirements.txt`
- **Step5:** Run `app.py`
