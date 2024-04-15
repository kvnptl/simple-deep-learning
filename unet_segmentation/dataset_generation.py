import tensorflow as tf
import numpy as np
np.random.seed(seed=9)
from typing import Tuple
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
import matplotlib

from simple_deep_learning.mnist_extended.mnist import display_digits
from simple_deep_learning.mnist_extended.semantic_segmentation import (create_semantic_segmentation_dataset, display_segmented_image, display_grayscale_array, plot_class_masks)

def mnist_extended_dataset(total_train_samples: int = 100, total_test_samples: int = 10, num_classes: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=total_train_samples, num_test_samples=total_test_samples, image_shape=(60, 60), num_classes=num_classes)

    return train_x, train_y, test_x, test_y

def display_segmented_image(y: np.ndarray, threshold: float = 0.5,
                            input_image: np.ndarray = None,
                            alpha_input_image: float = 0.2,
                            title: str = '',
                            ax: matplotlib.axes.Axes = None) -> None:
    """Display segemented image.

    This function displays the image where each class is shown in particular color.
    This is useful for getting a rapid view of the performance of the model
    on a few examples.

    Parameters:
        y: The array containing the prediction.
            Must be of shape (image_shape, num_classes)
        threshold: The threshold used on the predictions.
        input_image: If provided, display the input image in black.
        alpha_input_image: If an input_image is provided, the transparency of
            the input_image.
    """
    ax = ax or plt.gca()

    base_array = np.ones(
        (y.shape[0], y.shape[1], 3)) * 1
    legend_handles = []

    for i in range(y.shape[-1]):
        # Retrieve a color (without the transparency value).
        colour = plt.cm.jet(i / y.shape[-1])[:-1]
        base_array[y[..., i] > threshold] = colour
        legend_handles.append(mpatches.Patch(color=colour, label=str(i)))

    # plt.figure(figsize=figsize)
    # ax.imshow(base_array)
    # ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc='upper left')
    # ax.set_yticks([])
    # ax.set_xticks([])
    # ax.set_title(title)

    # if input_image is not None:
    #     ax.imshow(input_image[..., 0],
    #                cmap=plt.cm.binary, alpha=alpha_input_image)

    # if not ax:
    #     plt.show()

    return base_array

def plot_class_masks(y_true: np.ndarray, y_predicted: np.ndarray = None, title='') -> None:
    """Plot a particular view of the true vs predicted segmentation.

    This function separates each class into its own image and
    does not perform any thresholding.

    Parameters:
        y_true: True segmentation (image_shape, num_classes).
        y_predicted: Predicted segmentation (image_shape, num_classes).
            If y_predicted is not provided, only the true values are displayed.
    """
    num_rows = 2 if y_predicted is not None else 1

    num_classes = y_true.shape[-1]
    fig, axes = plt.subplots(num_rows, num_classes, figsize=(num_classes * 4, num_rows * 4))
    axes = axes.reshape(-1, num_classes)
    fig.suptitle(title)
    plt.tight_layout()

    for label in range(num_classes):
        axes[0, label].imshow(y_true[..., label], cmap=plt.cm.binary)
        axes[0, label].axes.set_yticks([])
        axes[0, label].axes.set_xticks([])

        if label == 0:
            axes[0, label].set_ylabel(f'Target')

        if y_predicted is not None:
            if label == 0:
                axes[1, label].set_ylabel(f'Predicted')

            axes[1, label].imshow(y_predicted[..., label], cmap=plt.cm.binary)
            axes[1, label].set_xlabel(f'Label: {label}')
            axes[1, label].axes.set_yticks([])
            axes[1, label].axes.set_xticks([])
        else:
            axes[0, label].set_xlabel(f'Label: {label}')

    plt.show()