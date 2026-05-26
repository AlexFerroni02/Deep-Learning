# 🌍 Deep Learning & Computer Vision: Satellite Imagery Analysis (xView)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Object Detection](https://img.shields.io/badge/Object%20Detection-YOLO%20%7C%20SSD%20%7C%20R--CNN-orange?style=for-the-badge)](https://github.com/AlexFerroni02/Deep-Learning)
[![Arch Evolution](https://img.shields.io/badge/Arch%20Evolution-FFNN%20%E2%96%B6%20CNN%20%E2%96%B6%20Obj%20Detect-brightgreen?style=for-the-badge)](https://github.com/AlexFerroni02/Deep-Learning)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

Welcome to this comprehensive repository chronicling a progressive deep learning journey focused on satellite imagery analysis using the **xView dataset**. This project represents a systematic, step-by-step exploration starting from foundational neural network concepts and advancing to complex, state-of-the-art computer vision tasks like image classification and object detection using **TensorFlow/Keras**.

The core focus of this project is to implement and study the **historic evolution of computer vision architectures**: starting from basic **Feed-Forward Neural Networks (MLP)**, transitioning to deep **Convolutional Neural Networks (CNN)** to leverage spatial invariance and local feature extraction, and culminating in advanced **Object Detection** frameworks (benchmarking One-Stage SSD/YOLO-style detectors against Two-Stage Faster R-CNN structures).

This pipeline tackles critical, real-world geospatial analysis challenges, such as severe class imbalances, high-resolution spatial dimensions, and localized object recognition (e.g., small cars, buses, trucks, and buildings).

---

## 🚀 Key Engineering & Architectural Achievements

Here are the key technical challenges solved in this repository:
- **Architecture Evolution & Benchmarking**: Designed and evaluated the full chronological pipeline of vision models, comparing how MLPs, deep custom CNNs, sliding windows, and deep modern detectors handle complex spatial features.
- **Tackled Extreme Class Imbalance**: Implemented a custom `CategoricalFocalCrossentropy` loss function dynamically weighted based on inverse class frequencies, resolving the severe class skew in xView classification (maximizing CNN validation accuracy to **~64.27%**).
- **Efficient Memory Pipelines**: Authored a custom, thread-safe Python generator (`generator_images`) performing on-the-fly mean/std normalization and real-time data augmentations to bypass disk bottlenecks and allow training on limited hardware resources.
- **Geospatial Coordinate Conversion**: Designed pipelines leveraging `rasterio` and OpenCV to transform multispectral high-resolution `.tif` geospatial files into standard RGB arrays, handling projection coordinate spaces.
- **Detector Architectural Benchmarking**: Implemented and benchmarked sliding-window baseline, One-stage (YOLO/SSD-inspired), and Two-stage (Faster R-CNN logic) region-proposal models to detect small vehicles, buildings, and trucks.

---

## 📚 Project Structure & Interactive Map

Below is a roadmap of the experiments, demonstrating a clear progression in complexity. Click on the folder names to navigate directly to their notebooks:

| Module / Stage | Description | Folder Link | Key Technologies |
| :--- | :--- | :--- | :--- |
| **1. FFNN Foundations** | Multi-Layer Perceptrons, activation functions, and baseline training loops. | [`FFNN/`](file:///c:/Users/alexf/OneDrive/Desktop/UPM/Deep%20Learning/Assignment/Deep-Learning/FFNN) | `TensorFlow`, `Keras`, `NumPy` |
| **2. Regularization & Optim** | Combating overfitting via weight decay, dropout, and dynamic callbacks. | [`FFNN_Reg/`](file:///c:/Users/alexf/OneDrive/Desktop/UPM/Deep%20Learning/Assignment/Deep-Learning/FFNN_Reg) | `EarlyStopping`, `ReduceLROnPlateau` |
| **3. Convolutional Networks** | Deep CNN blocks, Batch Normalization, Custom Generators, and Focal Loss. | [`CNN/`](file:///c:/Users/alexf/OneDrive/Desktop/UPM/Deep%20Learning/Assignment/Deep-Learning/CNN) | `Conv2D`, `ELU`, `Focal Loss` |
| **4. Object Detection** | Geospatial raster parsing, sliding window, SSD/YOLO, and Faster R-CNN models. | [`Object_Detection/`](file:///c:/Users/alexf/OneDrive/Desktop/UPM/Deep%20Learning/Assignment/Deep-Learning/Object_Detection) | `rasterio`, `OpenCV`, `Region Proposals` |

---

## 🧠 Deep-Dive Modular breakdown

### 1. Feed Forward Neural Networks (FFNN)
* **Directory:** [`FFNN/`](file:///c:/Users/alexf/OneDrive/Desktop/UPM/Deep%20Learning/Assignment/Deep-Learning/FFNN) (Experiments 1–4)
* **Objective:** Establish baseline models and confirm that the dataset loading pipelines function correctly.
* **Key Implementations:**
  - Designed Multi-Layer Perceptron (MLP) architectures to act as a baseline.
  - Explored activation functions (`ReLU`, `Sigmoid`, `Softmax`) on flattened image data.
  - Implemented standard forward propagation and backpropagation mechanisms.
  - Evaluated baseline loss functions (`CategoricalCrossentropy`) and basic training loops.

### 2. Regularization & Optimization (FFNN_Reg)
* **Directory:** [`FFNN_Reg/`](file:///c:/Users/alexf/OneDrive/Desktop/UPM/Deep%20Learning/Assignment/Deep-Learning/FFNN_Reg) (Experiments 1–9)
* **Objective:** Mitigate overfitting as network capacity scales up.
* **Key Implementations:**
  - **Weight Decay**: Applied L1 & L2 regularization to dense layers to prevent network weights from exploding.
  - **Dropout**: Tuned dropout rates dynamically to reduce co-adaptation between neurons.
  - **Dynamic Learning Schedulers**: Integrated `ReduceLROnPlateau` to scale down learning rates during training plateaus, smoothing validation loss convergence.
  - **EarlyStopping**: Halts training run automatically when validation metrics stop improving.

### 3. Convolutional Neural Networks (CNN)
* **Directory:** [`CNN/`](file:///c:/Users/alexf/OneDrive/Desktop/UPM/Deep%20Learning/Assignment/Deep-Learning/CNN) (Experiments 1–10)
* **Objective:** Capture spatial dependencies and visual hierarchies using deep 2D architectures.
* **Key Implementations:**
  - **On-the-Fly Augmentation**: Used `RandomFlip`, `RandomRotation`, and `RandomZoom` layers to address varying satellite imagery orientations.
  - **Robust Conv Blocks**: Chained multiple `Conv2D` layers paired with `BatchNormalization` (for convergence speed) and `ELU` activation (mitigating dying neurons).
  - **Memory-Efficient Data Generator**: Implemented a custom generator (`generator_images`) performing real-time scaling, mean/std normalization, and batch-wise loading to prevent Out-Of-Memory (OOM) errors.
  - **Advanced Loss Mechanics**: Introduced `CategoricalFocalCrossentropy` to down-weight easy-to-classify background tiles and force the network to learn scarce classes.
  - **Performance**: Achieved a peak validation accuracy of **~64.27%** around epoch 121 with our custom CNN architecture.

### 4. Object Detection
* **Directory:** [`Object_Detection/`](file:///c:/Users/alexf/OneDrive/Desktop/UPM/Deep%20Learning/Assignment/Deep-Learning/Object_Detection) (Base Model + Experiments 1–4)
* **Objective:** Localize and classify specific objects (**Small cars, Buses, Trucks, and Buildings**) in large satellite scenes.
* **Key Implementations:**
  - **Geospatial Preprocessing**: Leveraged `rasterio` via custom `load_geoimage` helpers to load `.tif` files, unpack multiband raster channels, and convert them to standard arrays.
  - **Annotation Dataclasses**: Structured bounding box coordinates, labels, and file paths using a custom Python `GenericImage` dataclass.
  - **Architectural Paradigms Evaluated**:
    1. **Sliding Window Baseline**: An exhaustive, sliding grid search to extract candidate crops for the classification network.
    2. **One-Stage Detector (YOLO/SSD style)**: Formulated object localization and classification as a single regression task to optimize for speed.
    3. **Two-Stage Detector (Faster R-CNN style)**: Created regional proposal networks (RPNs) that proposed bounding-box regions before feeding them into final classification heads, maximizing localization accuracy.

---

## 📊 Visualized Results

### CNN Performance: Confusion Matrix
The following confusion matrix shows the final custom CNN model's performance across the various xView classes, highlighting where spatial confusion occurs (e.g., distinguishing between trucks and buses under steep satellite angles):

![Custom CNN Confusion Matrix](file:///c:/Users/alexf/OneDrive/Desktop/UPM/Deep%20Learning/Assignment/Deep-Learning/CNN/resnet_confusion.png)

---

## 🛠 Getting Started

To reproduce the experiments or explore the notebooks, follow the instructions below:

### 1. Prerequisites & Environment
Ensure you have Python 3.8+ and a GPU-enabled environment (highly recommended for deep learning execution).

### 2. Clone the Repository
```bash
git clone <repository-url>
cd Deep-Learning
```

### 3. Install Dependencies
```bash
pip install tensorflow numpy rasterio opencv-python scikit-learn matplotlib
```
> **Note:** File paths in the Jupyter Notebooks (e.g., pointing to `/kaggle/input/` or `../PROJECT/xview_detection/`) are configured for absolute paths and cloud environments. Make sure to adjust them if running locally.

### 4. Run the Experiments
Launch Jupyter Notebooks to explore the step-by-step notebooks:
```bash
jupyter notebook
```

---

## 👨‍💻 Author

**Alex Ferroni**  
*AI & Deployment Engineer*  
* [LinkedIn](https://www.linkedin.com/in/alex-ferroni/) *(Insert your link here!)*
* [Portfolio/Website](https://alexferroni.github.io/) *(Insert your link here!)*
* Email: [alex.ferroni@example.com](mailto:alex.ferroni@example.com) *(Insert your email here!)*
