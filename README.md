# Diabetic Retinopathy Screening using CNN (ResNet18)

## Project Overview
This project implements a Convolutional Neural Network (CNN) based on ResNet18 architecture to screen and classify diabetic retinopathy from retinal images. The model categorizes the severity of diabetic retinopathy into five classes: Mild, Moderate, No DR (No Diabetic Retinopathy), Proliferate DR, and Severe.

## Data Source
The dataset used in this project is from the [Diabetic Retinopathy Detection competition on Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data).

## Dataset Distribution
The training dataset consists of retinal images distributed across classes as follows:

| Class          | Number of Images |
|----------------|------------------|
| Mild           | 370             |
| Moderate       | 999             |
| No DR          | 1805            |
| Proliferate DR | 295             |
| Severe         | 193             |

## Requirements
```python```
# Key dependencies
```tensorflow>=2.0.0```

```Keras```

```opencv-python```

```NumPy```

```matplotlib```

```seaborn```

```pandas```

```scikit-learn```

## Model Performance

### Accuracy Metrics
| Metric | Score |
|--------|-------|
| Training Accuracy | 84.38% |
| Testing Accuracy | 84.17% |
| Loss | 0.4812 |

### Confusion Matrix Analysis
| Actual/Predicted | Mild (0) | Moderate (1) | No DR (2) | Proliferate DR (3) | Severe (4) |
|-----------------|-----------|--------------|-----------|-------------------|------------|
| Mild (0)        | 43        | 13          | 8         | 6                | 1          |
| Moderate (1)    | 6         | 160         | 12        | 12               | 1          |
| No DR (2)       | 2         | 2           | 370       | 0                | 0          |
| Proliferate (3) | 1         | 17          | 3         | 30               | 1          |
| Severe (4)      | 1         | 23          | 2         | 5                | 14         |

#### Key Observations from Confusion Matrix:
1. Best Performance:
   - The "No DR" class shows the highest accuracy, with 370 correct predictions
   - "Moderate" class performs well with 160 correct classifications

2. Main Challenges:
   - Some confusion between Mild and Moderate cases (13 cases)
   - Proliferate DR shows some misclassification as Moderate (17 cases)
   - Severe cases are sometimes misclassified as Moderate (23 cases)

3. Class-wise Accuracy:
   | Class          | Accuracy | Total Cases |
   |----------------|----------|-------------|
   | Mild           | 60.56%   | 71         |
   | Moderate       | 83.77%   | 191        |
   | No DR          | 98.93%   | 374        |
   | Proliferate DR | 57.69%   | 52         |
   | Severe         | 31.11%   | 45         |

## Technical Implementation
### Architecture
- Base Model: ResNet18 (Pre-trained)
- Output Classes: 5 (Mild, Moderate, No DR, Proliferate DR, Severe)
- Weights File: `retina_weights.hdf5`

### Training Parameters
- Batch Size: 32
- Processing Time: ~42s/step
- Loss Function: Categorical Cross-entropy
- Optimizer: Adam

### Technologies Used
- TensorFlow/Keras for deep learning
- OpenCV for image processing
- Matplotlib and Seaborn for visualizations
- NumPy for numerical computations
- Pandas for data manipulation
