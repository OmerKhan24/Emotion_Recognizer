# Emotion Recognition

This project is a simple implementation of an emotion recognition system using Convolutional Neural Networks (CNNs) with Keras. The model is trained to recognize seven different emotions from grayscale images of faces. The emotions recognized are: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Overview

The project consists of two main parts:
1. **Model Training:** Training a CNN model on a dataset of face images labeled with different emotions.
2. **Real-time Emotion Recognition:** Using the trained model to predict emotions in real-time using a webcam.

## Dataset

The dataset used for training the model is the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) available on Kaggle. This dataset contains images of faces labeled with different emotions.

## How It Works

1. **Model Training:**
    - The training script loads the face images and their labels.
    - It preprocesses the images by converting them to grayscale and resizing them to 48x48 pixels.
    - The CNN model is built using several convolutional layers, max-pooling layers, dropout layers, and dense layers.
    - The model is trained on the preprocessed images and their corresponding labels.
    - The trained model is saved for later use.

2. **Real-time Emotion Recognition:**
    - The trained model is loaded.
    - The webcam captures video frames.
    - Faces are detected in each frame using OpenCV's Haar Cascade Classifier.
    - Each detected face is preprocessed and passed through the trained model to predict the emotion.
    - The predicted emotion is displayed on the video frame in real-time.

## Setup Instructions

Follow these steps to set up and run the project on your PC:

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

### Step-by-Step Setup

1. **Clone the Repository (optional):**
    ```bash
    git clone https://github.com/yourusername/emotion-recognition.git
    cd emotion-recognition
    ```

2. **Install the Required Packages:**
    ```bash
    pip install keras tensorflow pandas numpy scikit-learn tqdm matplotlib opencv-python
    ```

3. **Download the Dataset:**
    - Download the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) from Kaggle.
    - Extract the dataset and place the `train` and `test` directories inside an `images` directory in the project root.

4. **Train the Model:**
    - Run the `main.py` script to train the model.
    ```bash
    python main.py
    ```
    - The trained model and its architecture will be saved as `driver_model.h5` and `driver_model.json` respectively.

5. **Test the Model:**
    - You can test the trained model using some test images.
    - Modify the testing section in `mainv2.py` with your test images and run it.

6. **Run the Real-time Emotion Recognition:**
    - Run the `driver.py` script to start the webcam-based real-time emotion recognition.
    ```bash
    python driver.py
    ```

## Files and Directories

- `main.py`: Script for training the emotion recognition model.
- `driver.py`: Script for real-time emotion recognition using webcam.
- `images/`: Directory containing the training and test images.
- `driver_model.h5`: Trained model weights.
- `driver_model.json`: Model architecture in JSON format.

## Acknowledgments

- The project uses the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) available on Kaggle.
- This project was built using Keras, TensorFlow, OpenCV, and other open-source libraries.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/OmerKhan24/Emotion_Recognizer/blob/main/LICENSE) file for details.

