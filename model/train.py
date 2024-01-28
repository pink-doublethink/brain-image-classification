import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import *#build_model
from dataset import *# get_files, file_loader
from utils import *# save_model, save_plots
from model import *# pretrained_model


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e',
    '--epochs',
    type = int, 
    default = 20,
    help = 'Number of epochs to train network'
)
parser.add_argument(
    '-lr',
    '--learning_rate',
    type = float,
    dest = 'learning_rate',
    default = 0.0001,
    help = 'Learning rate for training the model'
)

args = vars(parser.parse_args())


# Training function

def train(model, trainloader, optimizer, criterion):
    model.train()
    print("Training...")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    
    for i, data in tqdm(enumerate(trainloader), total = len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # Forward prop
        outputs = model(image)
        
        # Calculating loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        
        # Calculating accuracy
        _, prediction = torch.max(outputs.data, 1)
        train_running_correct += (prediction == labels).sum().item()
        
        # Backward prop
        loss.backward()

        # Update weights
        optimizer.step()
        
    epoch_loss = train_running_loss / counter
    epoch_accuracy = 100 * (train_running_correct / len(trainloader.dataset))
    
    return epoch_loss, epoch_accuracy


# Validation function

def validate(model, testloader, criterion):
    model.eval()
    print('validation...')
    validation_running_loss = 0.0
    validation_running_correct = 0
    counter = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total = len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # Forward prop
            outputs = model(image)
            
            # Calculating loss
            loss = criterion(outputs, labels)
            validation_running_loss += loss.item()
            
            # Calculating accuracy
            _, prediction = torch.max(outputs.data, 1)
            validation_running_correct += (prediction == labels).sum().item()
            
    epoch_loss = validation_running_loss / counter
    epoch_accuracy = 100 * (validation_running_correct / len(testloader.dataset))
    
    return epoch_loss, epoch_accuracy


if __name__ == '__main__':
    # Load training and validation datasets
    train_set, validation_set, dataset_classes = get_files()
    print(f"Number of training images: {len(train_set)}")
    print(f"Number of validation images: {len(validation_set)}")
    print(f"Data classes: {dataset_classes}\n")
    
    # Load training and validation data loaders
    train_loader, validation_loader = file_loader(train_set, validation_set)
    
    # Learning parameters
    learning_rate = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}\n")
    
    model = build_model(
        pretrained = True,
        fine_tune = True,
        num_classes = len(dataset_classes)
    ).to(device)
    
    
    
    # Total parameters and trainable parameters
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"{total_parameters:,} parameters.")
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{trainable_parameters:,} training parameters.")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Tracking loss and accuracy
    train_loss, validation_loss = [], []
    training_accuracy, validation_accuracy = [], []
    
    # Train
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_loss_per_epoch, train_accuracy_per_epoch = train(model,
                                                               train_loader,
                                                               optimizer,
                                                               criterion)
        val_loss_per_epoch, val_accuracy_per_epoch = validate(model,
                                                              validation_loader,
                                                              criterion)
        train_loss.append(train_loss_per_epoch)
        validation_loss.append(val_loss_per_epoch)
        training_accuracy.append(train_accuracy_per_epoch)
        validation_accuracy.append(val_accuracy_per_epoch)
        print(f"Training loss: {train_loss_per_epoch:.3f}, Training accuracy: {train_accuracy_per_epoch:.3f}")
        print(f"Validation loss: {val_loss_per_epoch:.3f}, Validation accuracy: {val_accuracy_per_epoch:.3f}")
        #print("-.-" * 10)
        print("~ " * 100)
        time.sleep(3)
        
    # Save model
    save_model(epochs, model, optimizer, criterion)
    # Save loss and accuracy plots
    save_plots(training_accuracy, validation_accuracy, train_loss, validation_loss)
    print("Done")
    
    
    
# To run the model navigate to model directory and run "python train.py --epochs 10" ('insert_num_epochs') you can change the num