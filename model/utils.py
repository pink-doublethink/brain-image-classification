import torch
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def save_model(epochs, model, optimizer, criterion):
    torch.save({
        'epoch' : epochs,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : criterion,
    }, f"../outputs/model.pth")
    
    
def save_plots(train_accuracy, validation_accuracy, train_loss, validation_loss):
    # Accuracy plot
    plt.figure(figsize = (12, 10))
    plt.plot(train_accuracy, color = 'black', label = 'training accuracy')
    plt.plot(validation_accuracy, color = 'blue', label = 'validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"../outputs/accuracy.png")
    
    # Loss plot
    plt.figure(figsize = (12, 10))
    plt.plot(train_loss, color = 'black', label = 'training loss')
    plt.plot(validation_loss, color = 'blue', label = 'validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"../outputs/loss.png")