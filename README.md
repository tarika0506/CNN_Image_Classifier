Overview:
  This project focuses on classifying images from the Intel Image dataset into six categories: buildings, forest, glacier, mountain, sea, and street. It shows the progression from a basic CNN, which serves as a starting point, to an enhanced CNN that includes improved feature extraction and regularization techniques. The project also uses transfer learning models, which take advantage of pre-trained neural networks to improve classification accuracy and reduce training time.

Technologies Used:
  TensorFlow: An open-source framework for building and training neural networks.
  Keras: A high-level API in TensorFlow that simplifies defining and training deep learning models.
  Python libraries: numpy, matplotlib, seaborn, scikit-learn, pillow

Project Structure:
  Dataset Verification and Visualization
    Checks if the dataset folders and class subfolders exist.
    Displays sample images from each category for inspection.

  Basic CNN Model:
    Two or three convolutional layers followed by max pooling and dense layers.
    Serves as a baseline model, but lacks batch normalization and sufficient regularization.

  Enhanced CNN Model:
    Includes additional convolutional layers.
    Uses batch normalization to improve training stability.
    Adds dropout layers to prevent overfitting.
    Applies data augmentation to create more diverse training data.

  Transfer Learning Models(attempted):
    Uses pre-trained networks like VGG16 and ResNet50.
    Leverages learned features to improve performance on limited datasets.

  Features:
    Automatic verification of dataset structure and error handling.
    Visualization of sample images by class.
    Multiple CNN architectures for experimentation.
    Tracking and storing training history.
    Plots of training and validation accuracy for all models
