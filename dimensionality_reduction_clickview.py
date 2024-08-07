import os
import sys
from matplotlib import pyplot as plt
from medmnist import INFO
import numpy as np
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import torch
from PIL import Image

from utils import COLORS, DIMENSIONALITY_REDUCTION_SAMPLES, OUT_PATH, MedMNISTCategory, get_feats


def perform_feature_analysis_view(
    train_feats_data,
    test_feats_data,
    test_labels,
    test_data,
    data_flag,
    legend=True,
    explore_tsne_only=True,
):
    train_feats = get_feats(train_feats_data)
    test_feats = get_feats(test_feats_data)

    # In SimCLR pretraining we used a batch size of 128 and features = size*4
    assert train_feats.shape[1] == 512
    assert test_feats.shape[1] == 512

    os.makedirs(OUT_PATH, exist_ok=True)

    if explore_tsne_only:

        # Perform t-SNE
        test_feats_reduced = perform_tsne(
            train_feats,
            test_feats,
            perplexity=5,
        )

        plot_reduced_feats_view(
            test_feats_reduced,
            test_labels,
            test_data,
            data_flag,
            legend,
            # filter_indices=[5],
            #component_label="component",
        )

        fig_name = f"tsne-{data_flag}.png"
        plt.savefig(os.path.join(OUT_PATH, fig_name))

def perform_tsne(train_feats,test_feats, perplexity):
    # "It is highly recommended to use another dimensionality reduction method
    # (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the
    # number of dimensions to a reasonable amount (e.g. 50) if the number of
    # features is very high."
    # 
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    pca = decomposition.PCA(n_components=50)
    pca.fit(train_feats)

    feats_pca = pca.transform(test_feats)

    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        init="random",
        # Alternative: https://arxiv.org/pdf/1708.03229.pdf
        perplexity=perplexity,
    )

    return tsne.fit_transform(feats_pca)

def plot_reduced_feats_view(reduced_data, labels, image_data, title, legend=True):

    plt.style.use("ggplot")  # Use stylish ggplot

    #label_to_index = {label: idx for idx, label in enumerate(labels_dict.keys())}
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')

    def onclick(event):
        # Compute distances and identify the closest point
        distances = np.sqrt((reduced_data[:, 0] - event.xdata) ** 2 + (reduced_data[:, 1] - event.ydata) ** 2)
        index = np.argmin(distances)
        # Handle tensor format and channel information
        image_tensor = image_data[index][0] if isinstance(image_data[index], tuple) else image_data[index]
        img_array = image_tensor.numpy()  # Convert tensor to numpy array
        if img_array.shape[0] == 3:  # Check if there are 3 channels
            img_array = img_array.transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        else:
            img_array = img_array.squeeze()  # Remove single-dimensional entries from the shape
        img = Image.fromarray(np.uint8(img_array * 255))  # Convert to an image
        img.show()

    # Connect the onclick event
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.title(title)
    if legend:
        plt.legend(*scatter.legend_elements(), title="Labels")
    plt.show()

def plot_reduced_feats(feats_reduced, labels, data_flag, legend=True, filter_indices=[], component_label="principle component"):
    labels_dict = INFO[data_flag]["label"]

    # Use stylish plots
    plt.style.use("ggplot")

    # Clear plot
    plt.clf()

    num_classes = len(labels_dict)

    # Plot the data points coloured by their labels
    colors = COLORS

    for i in range(num_classes):
        if i in filter_indices:
            continue

        curr_label = labels_dict[str(i)]

        # BloodMNIST: The original 3rd label is very long
        if i == 3 and data_flag == MedMNISTCategory.BLOOD.value:
            curr_label = "immature granulocytes"

        plt.scatter(
            feats_reduced[labels == i, 0],
            feats_reduced[labels == i, 1],
            color=colors[i],
            label=curr_label,
            alpha=0.75,
            # Default marker area: 50
            # Make it smaller
            s=16,
        )

        plt.xlabel(f"{component_label} 1")
        plt.ylabel(f"{component_label} 2")

    if legend:
        plt.legend()
