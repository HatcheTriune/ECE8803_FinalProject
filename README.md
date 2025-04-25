# ECE8803: Fund of Machine Learning Group Project
RETINAL BIOMARKER DETECTION USING MULTIMODAL DEEP LEARNING ON THE OLIVES DATASET
This project addresses a multi-label classification task using the [OLIVES Dataset](https://huggingface.co/datasets/gOLIVES/OLIVES_Dataset). Aiming to develop an effective multimodal approach to detecting biomarkers in retinal OCT images by combining image data with clinical measurements and examining how combining image features with clinical metrics enhances detection accuracy across different biomarkers.

## Dependencies
1. To install the Python dependencies, assuming you already have a Python environment
You can install all required dependencies using:
```bash
pip install -r requirements.txt
```
2. Some other requirements
- CUDA-compatible GPU (really slow if you are using a CPU)
- Internet connection for dataset download 


## How to Run
```bash
python olives_project_augmented.py
```
After run this file the dataset will be downloaded and model will start to train the data. Final result will be printed in the terminal as: \
<img width="387" alt="12" src="https://github.com/user-attachments/assets/b2afadd9-afd8-474e-9a4a-1b02f266c0dd" />

![result](https://github.com/user-attachments/assets/a8736830-41e8-407b-b405-a316bfcc490d)

## Methodology
Our approach involves:
- Data Processing: Preparing the OLIVES dataset with appropriate filtering, normalization, and augmentation
- Feature Extraction:
   - Image features via pre-trained ResNet50 architecture
   - Clinical features (BCVA and CST) processed through a dedicated MLP
- Multimodal Fusion: Combining both feature types through a carefully designed fusion network
- Model Training: Implementing binary cross-entropy loss optimization with early stopping
- Evaluation: Assessing model performance using accuracy and F1 scores across all biomarkers
