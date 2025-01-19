import torch
import timm
import torch.nn as nn


def load_model(model_name, only_backbone=False):
    import torchvision.models as thmodels

    if model_name.startswith("resnet") and model_name.endswith("modified"):
        # model = timm.create_model(model_name.split("_modified")[0], pretrained=False)
        model = thmodels.__dict__[model_name.split("_modified")[0]](pretrained=False)
        model = prune_resnet(model)
    else:
        try:
            model = thmodels.__dict__[model_name](pretrained=False)
        except:
            model = timm.create_model(model_name, pretrained=False)

    if only_backbone is True:
        return prune_head(model)
    else: return model

def prune_resnet(model):
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model


def prune_head(model):
    if hasattr(model, "fc"):
        feature_dim = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif hasattr(model, "classifier"):
        feature_dim = model.classifier[-1].in_features
        model.classifier = torch.nn.Identity()
    elif hasattr(model, "heads"):
        feature_dim = model.heads[-1].in_features
        model.heads = torch.nn.Identity()
    elif hasattr(model, "head"):
        feature_dim = model.head.in_features
        model.head = torch.nn.Identity()
    return model, feature_dim
