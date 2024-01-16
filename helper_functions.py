"""
helper_functions.py

This module contains a collection of helper functions 
that can be used to perform various tasks.
These functions are designed to simplify common operations and improve code reusability.
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
import math
import cv2


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
    

def plot_confusion_matrix(y_true, y_pred,
                          class_names,
                          figsize=(12, 10),
                          fontsize=8,
                          text_rotation=45,
                          normalize=False,
                          save_filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted', fontsize=fontsize)
    plt.ylabel('True', fontsize=fontsize)
    plt.xticks(rotation=text_rotation)



    if normalize:
        plt.title('Normalized Confusion Matrix', fontsize=fontsize)
    else:
        plt.title('Confusion Matrix', fontsize=fontsize)

    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
        
def compare_histories(original_history, new_history, initial_epochs=5):
    
    """
    Compare and visualize two TensorFlow model training histories using a detailed plot.

    This function takes two TensorFlow model History objects, typically representing the training histories of a model
    before and after fine-tuning. It combines these histories and creates a stylish plot to compare and visualize
    training and validation metrics, including accuracy and loss, before and after fine-tuning.

    Args:
    original_history (tf.keras.callbacks.History): The training history of the original model.
    new_history (tf.keras.callbacks.History): The training history of the fine-tuned model.
    initial_epochs (int, optional): The number of initial training epochs before fine-tuning (default is 5).

    Returns:
    None: Displays the comparison plot.

    Example Usage:
    compare_histories(original_history, fine_tuned_history, initial_epochs=5)

    The function compares the training and validation metrics (accuracy and loss) between the original and fine-tuned
    models. It visually shows the effects of fine-tuning on the model's performance and helps assess the impact of
    changes made during the fine-tuning process.

    The resulting plot offers insights into the impact of fine-tuning on your model's performance.

    Note:
    Make sure to provide valid TensorFlow History objects for 'original_history' and 'new_history'.
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Create a stylish plot
    plt.figure(figsize=(12, 8))
    plt.suptitle("BEFORE FINE TUNING AND AFTER FINE TUNING",
                 fontsize=16,
                 fontweight="bold",
                 y=1.03,
                 color="red")

    # Training and Validation Accuracy
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy', color='green')
    plt.plot(total_val_acc, label='Validation Accuracy', color='blue')
    plt.axvline(initial_epochs, color='gray', linestyle='--', label='Start Fine Tuning', linewidth=2)
    plt.xticks(np.arange(1, len(total_acc),1))
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Training and Validation Loss
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss', color='red')
    plt.plot(total_val_loss, label='Validation Loss', color='purple')
    plt.axvline(initial_epochs, color='gray', linestyle='--', label='Start Fine Tuning', linewidth=2)
    plt.xticks(np.arange(1, len(total_acc),1))
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()
    plt.show()


def visualize_image_pairs(dataframe, num_pairs=5):

    """
    
    Visualizes pairs of images based on their ground truth and predicted labels from a DataFrame.

    Args:
    dataframe (pd.DataFrame): A DataFrame containing columns: 'File Path', 'Ground Truth Label', 'Predicted Label'.
    num_pairs (int, optional): The number of image pairs to visualize. Default is 5.

    Returns:
    None: Displays image pairs using Matplotlib.

    Example Usage:
    visualize_image_pairs(sorted_df, num_pairs=5)

    This function filters the input DataFrame to include only rows with different ground truth and predicted labels. It then randomly selects a specified number of image pairs and displays them in subplots. Each pair consists of an original image based on the ground truth label and a random predicted image based on the predicted label.

    The 'File Path' column in the DataFrame should contain the file paths to the images, 'Ground Truth Label' represents the true labels, and 'Predicted Label' contains the predicted labels.

    """

    # Filter the DataFrame to include only rows with different labels
    different_labels_df = dataframe[dataframe['Ground Truth Label'] != dataframe['Predicted Label']]

    # Randomly select rows from the filtered DataFrame
    random_rows = different_labels_df.sample(num_pairs)

    # Create subplots to display image pairs
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 15))

    for i, (_, row) in enumerate(random_rows.iterrows()):
        ground_truth_file_path = row['File Path']
        ground_truth_label = row['Ground Truth Label']
        predicted_label = row['Predicted Label']

        # Load and display the ground truth image
        ground_truth_image = plt.imread(ground_truth_file_path)
        axes[i, 0].imshow(ground_truth_image)
        axes[i, 0].set_title(f'Original: {ground_truth_label}',color = "green")
        axes[i, 0].axis('off')

        # Find a random predicted image based on the predicted label
        predicted_rows = different_labels_df[different_labels_df['Predicted Label'] == predicted_label]
        random_row = random.choice(predicted_rows.index)
        predicted_file_path = different_labels_df.loc[random_row, 'File Path']

        # Load and display the predicted image
        predicted_image = plt.imread(predicted_file_path)
        axes[i, 1].imshow(predicted_image)
        axes[i, 1].set_title(f'Predicted: {predicted_label},probability: {round(max_probability,2)}',color = "red")
        axes[i, 1].axis('off')

    plt.show()

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
    
    
# display image using openCV2

import cv2

def display_image(title, image):
    """
    Display an image with the given title.

    Parameters:
    - title (str): The title of the image window.
    - image (numpy.ndarray): The image to be displayed.

    Returns:
    - None
    """
    cv2.imshow(title, image)

    while True:
        # Wait for a key event with a delay of 1 millisecond
        key = cv2.waitKey(1) & 0xFF

        # Check if the pressed key is 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            break  # Exit the loop if 'q' is pressed

    # Explicitly wait for a short time before exiting the script
    cv2.waitKey(1)
