import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_brute_force_classification(model, image_path, nH=8, nW=8):
    """
    This function returns the probabilities of each window.
    Inputs:
        model: Model which is used
        image_path: path to the image to be analysed
        nH: number of windows in the vertical direction
        nW: number of windows in the horizontal direction
    Outputs:
        window_predictions: a (nH, nW, 3) np.array.
                            The last dim (size 3) is the probabilities
                            of each label (cat, dog, neg)
    """

    img = transform_img(image_path).to(device=device)

    ######### Your code starts here #########
    # Find the dimensions of each grid window using IMG_SIZE and nH, nW
    # Extract the window from the image with some amount of padding of your choice
    # Reshape the window to fit the input dimensions of the model and get the predicted output
    # Store the prediction in window_predictions and repeat across all number of windows
    window_height = IMG_SIZE // nH
    window_width = IMG_SIZE // nW
    window_predictions = np.zeros((nH, nW, 3))
    h_pad = window_height // 10
    w_pad = window_width // 10
    for i in range(nH):
        for j in range(nW):
            window = img[:, max(0, i * window_height - h_pad):min(IMG_SIZE, (i + 1) * window_height + h_pad), 
                            max(0, j * window_width - w_pad):min(IMG_SIZE, (j + 1) * window_width + h_pad)]
            window = window.unsqueeze(0)
            print(window.shape)
            window = nn.functional.interpolate(window, size=(299, 299), mode="bilinear", align_corners=False)
            with torch.no_grad():
                output = model(window)
                window_predictions[i, j] = output.cpu().numpy()

    ######### Your code ends here #########

    return window_predictions

def compute_convolutional_KxK_classification(model, image_path):
    """
    Computes probabilities for each window based on the convolution layer of Inception
    :param model: Combined model (base model + retrained classifier layer)
    :param image_path: Path to the image to be analyzed
    :return: KxK classification probabilities
    """
    img = transform_img(image_path).to(device=device).unsqueeze(0)

    base_model = model[0]
    classifier = model[1]

    children = list(base_model.children())[:-3]
    conv_model = nn.Sequential(*[child for child in children if not isinstance(child, nn.Module) or child.__class__.__name__ != 'InceptionAux'])
    conv_model.eval()

    ######### Your code starts here #########
    # Pass the image through the convolutional model
    # Get the output of the convolutional model
    # Reshape output to (1 * 8 * 8, 2048)
    # Run output through linear classifier

    with torch.no_grad():
        output = conv_model(img)
        _, C, K, K = output.shape
        output = output.view(1 * K * K, C)
        predictionsKxK = classifier(output)
        predictionsKxK = predictionsKxK.view(K, K, 3)

    ######### Your code ends here #########

    return predictionsKxK

def compute_and_plot_saliency(model, image_path):
    """
    This function computes and plots the saliency plot.
    You need to compute the matrix M detailed in section 3.1 in
    K. Simonyan, A. Vedaldi, and A. Zisserman,
    "Deep inside convolutional networks: Visualising imageclassification models and saliency maps,"
    2013, Available at https://arxiv.org/abs/1312.6034.

    :param model: Model which is used
    :param image_path: Path to the image to be analysed
    :return: None
    """
    img = transform_img(img_path=image_path).to(device=device).unsqueeze(0)
    img.requires_grad_(True)
    
    ######### Your code starts here #########

    # Forward pass to obtain logits
    # Compute the gradient of the top class logit with respect to the input image
    # Reshape the gradients to match the image dimensions
    # Handle potential multi-channel gradients (e.g., RGB)
    # Create the saliency map by taking the absolute value of the gradients

    logits = model(img)
    top_class = torch.argmax(logits, dim=1)
    top_class_score = logits[0, top_class]
    w = torch.autograd.grad(top_class_score, img)[0]
    M = w.abs().squeeze(0).max(dim=0)[0]

    ######### Your code ends here #########
    
    plt.subplot(2, 1, 1)
    plt.imshow(M.cpu().detach().numpy())
    plt.title("Saliency with respect to predicted class %s" % LABELS[top_class])
    plt.subplot(2, 1, 2)
    plt.imshow(decode_jpeg(image_path))
    plt.savefig("../plots/saliency.png")
    plt.show()

def plot_classification(image_path, classification_array, path):
    if classification_array is None:
       return
    nH, nW, _ = classification_array.shape
    image_data = decode_jpeg(image_path)
    if image_data is None:
       return
    aspect_ratio = float(image_data.shape[0]) / image_data.shape[1]
    plt.figure(figsize=(8, 8 * aspect_ratio))
    p1 = plt.subplot(2, 2, 1)
    plt.imshow(classification_array[:, :, 0], interpolation="none", cmap="jet")
    plt.title("%s probability" % LABELS[0])
    p1.set_aspect(aspect_ratio * nW / nH)
    plt.colorbar()
    p2 = plt.subplot(2, 2, 2)
    plt.imshow(classification_array[:, :, 1], interpolation="none", cmap="jet")
    plt.title("%s probability" % LABELS[1])
    p2.set_aspect(aspect_ratio * nW / nH)
    plt.colorbar()
    p2 = plt.subplot(2, 2, 3)
    plt.imshow(classification_array[:, :, 2], interpolation="none", cmap="jet")
    plt.title("%s probability" % LABELS[2])
    p2.set_aspect(aspect_ratio * nW / nH)
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(image_data)
    plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--scheme", type=str)
    FLAGS, _ = parser.parse_known_args()
    maybe_makedirs("../plots")

    model = torch.load("./trained_models/trained.pth").to(device)
    model.eval()
    
    if FLAGS.scheme == "brute":
        plot_classification(
            FLAGS.image,
            compute_brute_force_classification(model, FLAGS.image, 8, 8),
            "../plots/brute.png"
        )
    elif FLAGS.scheme == "conv":
        plot_classification(
            FLAGS.image,
            compute_convolutional_KxK_classification(model, FLAGS.image),
            "../plots/saliency.png"
        )
    elif FLAGS.scheme == "saliency":
       compute_and_plot_saliency(model, FLAGS.image)
    else:
        print("Unrecognized scheme:", FLAGS.scheme)