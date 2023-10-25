"""
helper_functions.py

This module contains a collection of helper functions that can be used to perform various tasks. These functions are designed to simplify common operations and improve code reusability.

Available functions:
- `function1(arg1, arg2)`: Description of what function1 does.
- `function2(arg1)`: Description of what function2 does.
- ...

Usage:
You can import and use the functions in this module as follows:

"""
# ALL DEPENDENCIES

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import requests
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow_hub as hub


# PLOTTING FUCTIONS

def plot_random_sample(directory):
    
    """
    Generate a random plot from a random image file in a randomly selected subdirectory.

    Parameters:
    - directory (str): The directory containing the subdirectories with the image files.

    Returns:
    - None: If no subdirectories or image files are found.

    This function searches for subdirectories within the specified directory and randomly selects one. It then selects a random image file from that subdirectory and plots it using matplotlib. The plot does not contain axes and displays the name of the subdirectory as the title.
    """
    
    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
    if not subdirectories:
        print("No subdirectories found in the specified directory.")
        return

    random_subdir = random.choice(subdirectories)
    subdir_path = os.path.join(directory, random_subdir)

    files = os.listdir(subdir_path)
    if not files:
        print(f"No files found in subdirectory: {random_subdir}")
        return

    random_file = random.choice(files)
    img = mpimg.imread(os.path.join(subdir_path, random_file))

    plt.imshow(img)
    plt.axis('off')
    plt.title(random_subdir.split('.')[0])
    plt.show()
    
    def create_model(model_url, num_classes=10):
        
        """
        Create a model using the given model URL and number of classes.
        
        Args:
            model_url (str): The URL of the model to use as the feature extractor layer.
            num_classes (int, optional): The number of classes for the output layer. Defaults to 10.
        
        Returns:
            tf.keras.Model: The created model.
        """
        
        feature_extractor_layer = hub.KerasLayer(model_url,
                                                 trainable=False,
                                                 name='feature_extraction_layer',
                                                 input_shape=(IMAGE_SIZE + (3,)))
        model = tf.keras.Sequential([
            feature_extractor_layer,
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model
    

def plot_curves(history,name_model):
    
    """
    Plots the training curves of a model.

    Args:
        history (History): The training history object that contains the accuracy and loss values.
        name_model (str): The name of the model.

    Returns:
        None
    """
    
    plt.figure(figsize=(12, 6))
    # set up a super title for the figure
    plt.suptitle(f'Training Curves of:\n{name_model}',
                 fontsize=16,
                 fontweight='bold',
                 y=1.02, color='red')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label=f'Training Accuracy: {acc[-1]:.3f}', color='blue', linestyle='-', marker='o')
    plt.plot(epochs,val_acc, label=f'Validation Accuracy: {val_acc[-1]:.3f}', color='green', linestyle='-', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs,loss, label=f'Training Loss: {loss[-1]:.3f}', color='red', linestyle='-', marker='o')
    plt.plot(epochs,val_loss, label=f'Validation Loss: {val_loss[-1]:.3f}', color='purple', linestyle='-', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def plot_random_images(N, directory):
    
    """
    Plots N random images from the specified directory.

    Parameters:
        N (int): The number of random images to plot.
        directory (str): The directory to search for image files.

    Returns:
        None
    """
    
    # Get a list of image files in the specified directory and its subdirectories
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append((os.path.join(root, file), root))

    if not image_files:
        print("No image files found in the specified directory.")
        return

    # Randomly select N unique images
    selected_images = random.sample(image_files, N)

    # Calculate the number of rows and columns for subplots
    num_rows = math.ceil(N / 5)
    num_cols = min(N, 5)

    # Create subplots for displaying the selected images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))
    axes = axes.ravel()  # Flatten the 2D array of axes

    for i, (image_path, subdirectory) in enumerate(selected_images):
        ax = axes[i]
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.axis('off')
        # Set the title with the shape and name of the subdirectory
        ax.set_title(f"Shape: {img.shape}\nSubdirectory: {os.path.basename(subdirectory)}")

    # Hide any empty subplots
    for i in range(N, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    
    
def plot_decision_bn(model,X,y):
    
    """
    Plots the decision boundaries for a binary or multiclass classification model.

    Parameters:
        model: The trained classification model.
        X: The input feature matrix.
        y: The target labels.

    Returns:
        None
    """
    
    X_min,X_max = X[:,0].min() - 0.1,X[:,0].max() + 0.1
    y_min,y_max = X[:,1].min() - 0.1,X[:,1].max() + 0.1
    xx,yy = np.meshgrid(np.linspace(X_min,X_max,100),
                        np.linspace(y_min,y_max,100))
    x_in = np.c_[xx.ravel(),yy.ravel()]
    y_pred = model.predict(x_in)
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        y_pred = np.argmax(y_pred,axis = 1).reshape(xx.shape)
    else:
        print("doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)
    plt.contourf(xx,yy,y_pred,cmap = plt.cm.RdYlBu,alpha = 0.7)
    plt.scatter(X[:,0],X[:,1],c = y,cmap = plt.cm.RdYlBu,s = 15)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    
# GET IMAGE FROM THE WEB:

def download_image(url):
    
    """
    Downloads an image from the given URL.
    
    Parameters:
        url (str): The URL of the image to download.
        
    Returns:
        bytes: The content of the downloaded image if the download is successful.
        
    Prints:
        str: "Failed to download the image. Check the URL." if the download fails.
    """
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print("Failed to download the image. Check the URL.")
        
        
def plot_confusion_matrix(y_true, y_pred):
    
    """
    Generate a confusion matrix plot based on the true labels and predicted labels.

    Parameters:
    - y_true: The true labels.
    - y_pred: The predicted labels.

    Returns:
    None
    """
    
    # Compute the confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')

    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # Show the plot
    plt.show()
    
def unzip_data(filename):
    
    """
    Extracts the contents of a zip file.

    Args:
        filename (str): The name of the zip file.

    Returns:
        None
        
    Notes: in Colab remember to use !wget to download the zip file first
    and after that unzip it.
    """
    
    import zipfile
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall()
    zip_ref.close()
    