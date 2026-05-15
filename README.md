# 🌍 Deep Learning & Computer Vision: xView Recognition & Detection

Welcome to this comprehensive repository chronicling a deep learning journey focused on satellite imagery analysis using the **xView dataset**. This project represents a progressive, step-by-step exploration starting from foundational neural network concepts and advancing to complex, state-of-the-art computer vision tasks like image classification and object detection using **TensorFlow/Keras**.

The core objective of this project is to tackle real-world challenges in satellite imagery, such as severe class imbalances, high resolution spatial data, and complex object recognition (e.g., Small cars, Buses, Trucks, Buildings).

---

## 📚 Table of Contents
1. [Project Overview & Tech Stack](#-project-overview--tech-stack)
2. [1. Feed Forward Neural Networks (FFNN)](#1-feed-forward-neural-networks-ffnn)
3. [2. Regularization & Optimization (FFNN_Reg)](#2-regularization--optimization-ffnn_reg)
4. [3. Convolutional Neural Networks (CNN)](#3-convolutional-neural-networks-cnn)
5. [4. Object Detection](#4-object-detection)
6. [Getting Started](#-getting-started)

---

## 🛠 Project Overview & Tech Stack

The repository is structured into four distinct modules, each representing a leap in architectural complexity and capability. The environment is fully configured to leverage GPU acceleration (e.g., CUDA, Kaggle environments).

**Key Technologies Used:**
- **Deep Learning Framework:** TensorFlow & Keras
- **Computer Vision & Image Processing:** OpenCV (`cv2`), Rasterio (for geospatial data), Pillow
- **Data Manipulation & Analysis:** NumPy, Pandas, scikit-learn
- **Visualization:** Matplotlib, Seaborn

---

## 1. Feed Forward Neural Networks (FFNN)
**Directory:** `FFNN/` | **Experiments:** `experiment_1` to `experiment_4`

This initial phase establishes the absolute basics of Deep Learning using Feed Forward Neural Networks (Multi-Layer Perceptrons) for basic image classification tasks.

**Key Explorations:**
- Designing the foundational architecture (Input layers, Dense hidden layers, Output layer).
- Exploring fundamental activation functions (`ReLU`, `Sigmoid`, `Softmax`).
- Implementing standard forward propagation and backpropagation mechanisms.
- Analyzing baseline loss functions (`CategoricalCrossentropy`) and basic training loops to establish a benchmark for future improvements.

---

## 2. Regularization & Optimization (FFNN_Reg)
**Directory:** `FFNN_Reg/` | **Experiments:** `experiment_1` to `experiment_9`

As network depth increases, so does the risk of overfitting, especially with complex datasets like xView. Across 9 distinct experiments, this module thoroughly investigates techniques to improve model generalization.

**Key Implementations:**
- **Weight Decay:** L1 & L2 Regularization applied to Dense layers to penalize large weights.
- **Dropout:** Strategically disabling neurons during training to prevent co-adaptation.
- **Advanced Callbacks:** Implementing `EarlyStopping` to halt training when validation loss degrades, and `ReduceLROnPlateau` to dynamically adjust the learning rate for smoother convergence.
- **Addressing Data Imbalance:** Early attempts at class weighting to ensure minority classes are not overwhelmed by majority classes.

---

## 3. Convolutional Neural Networks (CNN)
**Directory:** `CNN/` | **Experiments:** `experiment_1` to `experiment_10`

Transitioning from flat arrays to spatial 2D data, this module represents the core of the image recognition task. Over 10 iterative experiments, the architecture evolves into a highly optimized CNN tailored for satellite tiles.

**Architectural Innovations & Features:**
- **Data Augmentation:** Native Keras layers (`RandomFlip`, `RandomRotation`, `RandomZoom`) to artificially expand the dataset and improve robustness against varying satellite angles.
- **Deep Convolutional Blocks:** Multiple `Conv2D` layers paired with `BatchNormalization` (for training stability) and `ELU` (Exponential Linear Unit) activations to mitigate the dying ReLU problem.
- **Dimensionality Reduction:** Utilizing `MaxPooling2D` and `GlobalAveragePooling2D` to cleanly transition from feature maps to dense classification heads.
- **Custom Data Generators:** A robust `generator_images` pipeline that handles image loading, precomputed mean/std normalization, and one-hot label encoding on-the-fly.
- **Focal Loss:** Implementation of `CategoricalFocalCrossentropy` with a dynamically computed `alpha` list based on class weights. This was critical in addressing the extreme class imbalance in the xView recognition dataset.
- **Results:** Through rigorous hyperparameter tuning and utilizing callbacks (`TerminateOnNaN`, `ModelCheckpoint`), the network achieved a peak validation accuracy of **~64.27%** (converging around epoch 121 in advanced experiments).

---

## 4. Object Detection
**Directory:** `Object_Detection/` | **Experiments:** `Base_model` + `obj-detection1` to `obj-detection4`

The final and most advanced phase of the project. Moving beyond answering "what is in the image?" to solving "what is in the image, and exactly where is it?". The goal is to detect and localize **Small cars, Buses, Trucks, and Buildings**.

**Pipeline & Data Handling:**
- **Geospatial Processing:** Using `rasterio` via custom utilities (`load_geoimage`) to load large `.tif` files and convert spectral bands into OpenCV-compatible NumPy arrays.
- **Annotation Management:** A custom `GenericImage` dataclass built to seamlessly manage image metadata alongside bounding box coordinates and object labels.

**Architectures Explored:**
1. **Sliding Window Approach:** Establishing a baseline logic for localizing objects across large tiles.
2. **One-Stage Detectors (YOLO / SSD logic):** Framing detection as a single regression problem to predict bounding boxes and class probabilities simultaneously. This approach focuses on real-time inference speed.
3. **Two-Stage Detectors (Faster R-CNN logic):** Implementing a more complex pipeline that first proposes Regions of Interest (RoIs) and subsequently classifies and refines those specific regions, trading off computational speed for higher localization accuracy.

---

## 🚀 Getting Started

To reproduce the experiments or explore the notebooks, ensure your environment supports GPU execution (highly recommended).

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Deep-Learning
   ```

2. **Install Dependencies:**
   Ensure you have the required libraries installed:
   ```bash
   pip install tensorflow numpy rasterio opencv-python scikit-learn matplotlib
   ```
   *(Note: File paths in the notebooks, such as those pointing to `/kaggle/input/` or `../PROJECT/xview_detection/`, may require adjustment based on your local or cloud environment).*

3. **Explore the Notebooks:**
   Launch Jupyter to explore the iterative experiments:
   ```bash
   jupyter notebook
   ```

---
## 👨‍💻 Author
**Alex Ferroni**  
AI & Deployment Engineer
