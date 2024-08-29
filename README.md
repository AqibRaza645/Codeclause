Image Classification with TensorFlow
This project demonstrates how to build and train a convolutional neural network (CNN) for binary image classification using TensorFlow. The CNN model distinguishes between two classes of images. Below is a detailed overview of the code and the steps involved.

Data Loading and Visualization
The dataset is loaded using TensorFlow's image_dataset_from_directory function, which creates a dataset object from images stored in a directory structure. The dataset contains 2023 images divided into two classes.

We visualize a few sample images from the dataset to get an idea of the input data. The images are displayed using matplotlib, with each image's corresponding label shown as the title.

Data Preprocessing and Splitting
To prepare the data for training, we normalize the pixel values by scaling them to the range [0, 1]. The dataset is then shuffled and split into training, validation, and test sets. The split ratios are approximately 70% for training, 20% for validation, and 10% for testing. The sizes of these subsets are printed out for verification.

Model Architecture
The CNN model is defined using TensorFlow's Keras API. The architecture consists of:

Three Convolutional Layers: Each followed by a MaxPooling layer.
Flatten Layer: To convert the 2D feature maps into a 1D feature vector.
Dense Layers: Including one fully connected layer with 256 units and a final output layer with a single unit (sigmoid activation) for binary classification.
The model is compiled with the Adam optimizer and binary cross-entropy loss. Accuracy is used as the performance metric.

Model Training
The model is trained for 20 epochs on the training set, with validation performed on the validation set. Training progress is monitored through accuracy and loss metrics for both training and validation data.

Evaluation
The trained model is evaluated on the test set. Precision, Recall, and Accuracy metrics are calculated to assess the model's performance. The results indicate perfect scores (1.0) for all metrics, suggesting excellent performance on the test set.

Prediction
An image is read using OpenCV, resized to the input dimensions expected by the model (256x256 pixels), and normalized. The model's prediction is then obtained, and the class is determined based on a threshold of 0.5. The output indicates whether the image is classified as "Dog" or "Cat."
