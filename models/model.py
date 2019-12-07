import torch.nn as nn

from torchvision.models import resnet50, resnet101, resnet152
from pretrainedmodels.models import xception, inceptionv4

torchvision_models = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}
pretrained_models = {
    'xception': xception,
    'inceptionv4': inceptionv4
}


def get_model(model_type, num_classes, pretrained=True):

    global torchvision_models, pretrained_models
    if model_type in torchvision_models.keys():
        model = torchvision_models[model_type](pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_type in pretrained_models.keys():
        model = pretrained_models[model_type]
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)

    return model
