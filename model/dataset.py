from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Constants import *


# Training and Validation image preprocessing

def train_preprocessing(IMAGE_SIZE):
    train_augmenting = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.GaussianBlur(kernel_size = (5, 9),sigma = (0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor = 2, p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.465, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    
    return train_augmenting


def validation_preprocessing(IMAGE_SIZE):
    validation_augmenting = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.465, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    
    return validation_augmenting


# Loading files from directory

def get_files():
    train_set = datasets.ImageFolder(
        TRAIN_PATH,
        transform = (train_preprocessing(IMAGE_SIZE))
    )
    validation_set = datasets.ImageFolder(
        VALIDATION_PATH,
        transform = (validation_preprocessing(IMAGE_SIZE))
    )
    
    return train_set, validation_set, train_set.classes


def file_loader(train_set, validation_set):
    train_loader = DataLoader(
        train_set,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = NUM_WORKERS
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKERS
    )
    
    return train_loader, validation_loader