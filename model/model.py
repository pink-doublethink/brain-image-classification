import torchvision.models as models
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url



def build_model(pretrained = True, fine_tune = True, num_classes = 4):
    """
    Builds a ResNet model for image classification.

    Args:
        pretrained (bool): Whether to load pre-trained weights.
        fine_tune (bool): Whether to fine-tune all layers.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The built model.
    """

    # Load the pre-trained ResNet model
    model = models.resnet50(pretrained=pretrained)

    # Freeze all layers if fine-tuning is not enabled
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    # Change the final classification head to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model