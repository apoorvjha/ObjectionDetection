### README  

#### **Project Title**  
**Object Detection and Classification Using Deep Learning Models**

---

#### **Description**  
This project demonstrates the development and evaluation of deep learning models for object detection and classification across mixed indoor and outdoor environments. The models use the Caltech-101 dataset and aim to predict bounding boxes and classify images into object categories.  

---

#### **Requirements**  

**Python Version**: 3.8 or higher  

##### **Required Libraries**:  
You can install the required libraries by running:  
```bash
pip install -r requirements.txt
```  
---

#### **Dataset**  

The project uses the **Caltech-101 dataset**.  
1. Download the dataset from the [official Caltech-101 website](http://www.vision.caltech.edu/Image_Datasets/Caltech101/).  
2. Extract the dataset into a directory structure as follows:  
   ```plaintext
   caltech-101/
   ├──caltech-101/
    ├── 101_ObjectCategories/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    ├── Annotations/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── ...
   ```

---

#### **Code Organization**  

```plaintext
project/
├── caltech-101/               # Contains dataset
├── source/                    # Source code
│   ├── dataset.py             # Class to hold the dataset in pytorch compatible format
│   ├── live_demo.ipynb        # Notebook to load the best model (VGG-19) and perform inference and visualization
│   ├── main.py                # Main orchestrator code to call the training and evaluation methods.
│   ├── model_evaluator.py     # Model Evaluation pipeline
│   ├── model_trainer.py       # Model Training pipeline
│   ├── loss_function.py       # Custom loss function
│   └── utility.py             # Helper functions
│   └── model.py               # Model Architectures
│   └── runtime_parameters.py  # Parameters used across the codebase
├── requirements.txt           # Library dependencies
└── README.md                  # Project documentation
```

---

#### **How to Run the Code**  

1. **Clone the Repository**:  
   Clone this repository to your local machine.  
   ```bash
   git clone https://github.com/apoorvjha/ObjectionDetection
   cd ObjectDetection/
   ```

2. **Install Dependencies**:  
   Install the required Python packages:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models**:  
   Train all models by running the `main.py` script:  
   ```bash
   cd source && python main.py
   ```
   Alternatively, to train a specific model, update the configuration in `main.py` or call the corresponding function in `train.py`.

7. **Visualize Results**:  
   View training and evaluation metrics (precision, recall, F1 score, and bounding box accuracy) stored in the console output.

---

#### **Key Features**  
- **Models Implemented**:  
  - **VGG-19**  
  - **Custom CNN**  
  - **Fast R-CNN**  
  - **Vision Transformer**  

- **Metrics Evaluated**:  
  - Precision, Recall, F1 Score (Macro & Micro)  
  - Cosine Similarity of Bounding Boxes  
  - Intersection over Union (IoU)  

- **Preprocessing Techniques**:  
  - Grayscale Image Processing  
  - Data Augmentation (sharpening, smoothing)  

---

#### **Future Improvements**  
- Incorporate additional datasets for further model robustness.  
- Experiment with advanced architectures such as YOLO or Mask R-CNN.  
- Introduce hyperparameter tuning with grid search or Bayesian optimization.  

---

#### **Acknowledgments**  
This project was inspired by challenges in object detection for mixed environments, leveraging the Caltech-101 dataset as a foundational resource.  

#### **License**  
[MIT License](LICENSE)  
