import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def classify(model, test_dir, device):
    """
    Classifies all images in test_dir
    :param model: Model to be evaluated
    :param test_dir: Directory including the images
    :return: None
    """
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    ######### Your code starts here #########
    # Classify all images in the given folder, test_dir
    # Create the test_dataset and test_dataloader (with batch_size = 1, shuffle = False)
    # Calculate the accuracy and the number of test samples in the folder 
    # by passing the images into the model and comparing to the true labels
    
    # Also print out the img_paths of the incorrect classifications for future reference
    test_dataset = ImageDataset(img_dir=test_dir, labels=LABELS, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    num_test = 0
    correct = 0
    incorrect_img_paths = []
    for image, label, img_path in test_dataloader:
        if image is None:
            continue
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            num_test += 1
            if predicted == label:
                correct += 1
            else:
                incorrect_img_paths.append(img_path)
    accuracy = correct / num_test
    print(incorrect_img_paths)

    ######### Your code ends here #########

    print(f"Evaluated on {num_test} samples.")
    print(f"Accuracy: {accuracy * 100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_dir", type=str, default="datasets/test/")
    FLAGS, _ = parser.parse_known_args()

    #Load the model
    model = torch.load('./trained_models/trained.pth').to(device)
    model.eval()
    classify(model, FLAGS.test_image_dir, device)


