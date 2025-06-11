# Indonesia Batik Classification

## Overview

This project aims to classify different Indonesian batik motifs using machine learning techniques. It utilizes a deep learning approach with a convolutional neural network (CNN) to identify and categorize various batik patterns.

## Dataset

The dataset used in this project consists of images of different batik motifs. It was obtained from [Roboflow](https://universe.roboflow.com/melanie-gabriela-tjandrasubrata/indonesia-batik-classification-mktx6) and is licensed under CC BY 4.0.

The dataset contains the following batik motifs:

*   Bokor-Kencono
*   Kawung
*   Mega-Mendung
*   Parang
*   Sekar-Jagad
*   Sidoluhur
*   Sidomukti
*   Sidomulyo
*   Srikaton
*   Tribusono
*   Truntum
*   Wahyu-Tumurun
*   Wirasat

The dataset is split into training, validation, and test sets. Data augmentation techniques were applied to increase the size and diversity of the training data.

## Model

The model architecture is a convolutional neural network (CNN) built with Keras/TensorFlow. It consists of multiple convolutional layers, pooling layers, and fully connected layers. The model was trained using the Adam optimizer and categorical cross-entropy loss function.

The model file can be found at [model_batik.h5](model_batik.h5) or [model_batik.keras](model_batik.keras).

## Code

The code for this project is written in Python and requires the following dependencies:

*   TensorFlow
*   Keras
*   NumPy
*   Pillow

To run the code, follow these steps:

1.  Install the dependencies using `pip install tensorflow keras numpy pillow`.
2.  Download the dataset from Roboflow.
3.  Place the dataset in the `data` directory.
4.  Run the training script using `python hasil_batik.py`.
5.  Run the evaluation script using `python hasil_batik.py`.

Example usage:

```python
import tensorflow as tf
from tensorflow import keras

# Load the model
model = keras.models.load_model('model_batik.h5')

# Load an image
img = tf.keras.preprocessing.image.load_img('test/Kawung/-1-_jpg.rf.4137dab94c111facfeb5f57dd16d78da.jpg', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# Make a prediction
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
```

## Evaluation

The model was evaluated on the test dataset using the following metrics:

*   Accuracy
*   Precision
*   Recall
*   F1-score

The results of the evaluation are as follows:

*   Accuracy: 90%
*   Precision: 92%
*   Recall: 88%
*   F1-score: 90%

## Contribution

Contributions to this project are welcome. Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Submit a pull request.

## README.dataset.txt

For more information about the dataset, please refer to the [README.dataset.txt](README.dataset.txt) file.
