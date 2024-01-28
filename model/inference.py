import torch
import cv2
import numpy as np
import glob as glob
import os
from torchvision import transforms
from model import *
from Constants import *


class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Load saved model
model = build_model(pretrained = False, fine_tune = False, num_classes = 4)
checkpoint = torch.load("../outputs/model.pth", map_location = DEVICE)
#print("Loading saved model")
model.load_state_dict(checkpoint['model_state_dict'])

# Get test files
test_files = glob.glob(f"{TEST_PATH}/*")

# Iterate over images
for test_file in test_files:
    # Get true class name and make copy
    actual_class = test_file.split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(test_file)
    test_image = image.copy()
    
    # Preprocess images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.465, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(DEVICE)
    
    # Test on image
    outputs = model(image)
    outputs = outputs.detach().numpy()
    predicted_class = class_names[np.argmax(outputs[0])]
    print(f"Predicted: {predicted_class.lower()}\t Actual class: {actual_class}")
    
    # Inscribe the text onto the images
    cv2.putText(
        test_image,
        f"Actual: {actual_class}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 100, 225),
        1,
        lineType = cv2.LINE_AA
    )
    cv2.putText(
        test_image,
        f"Predicted: {predicted_class.lower()}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (65, 65, 155),
        1,
        lineType = cv2.LINE_AA
    )
    

    cv2.imwrite(f"../outputs/{actual_class}.png", test_image)
    
    
# Run with "python inference.py"