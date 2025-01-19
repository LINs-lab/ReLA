import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.datasets import VOCSegmentation
import torchvision.models.segmentation as models
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import os
from torchvision.models.segmentation import FCN
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models._utils import IntermediateLayerGetter
from utilities import get_lastdir, str_with_dashes, Logger
import statistics


def get_segment_args(args):
    if args.seed is None:
        args.seed = get_lastdir(args.root_directory)
    return args


def replace_pool_and_get_layers(model):
    """
    Replace the last pooling layer in the model with the layers before it
    and return the model and the output channels of the last conv layer.
    """
    last_conv_out_channels = None
    layers_to_return = []
    pooling_layer_name = None

    # Traverse the model to find the last convolutional layer and the last pooling layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_out_channels = module.out_channels
        elif isinstance(
            module,
            (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.MaxPool2d, nn.AvgPool2d),
        ):
            pooling_layer_name = name

    # If pooling layer is found, get all layers before it
    if pooling_layer_name:
        for name, module in model.named_children():
            if pooling_layer_name.startswith(name):
                break
            layers_to_return.append(name)

    # Use IntermediateLayerGetter to return the layers before the pooling layer
    if layers_to_return:
        model = IntermediateLayerGetter(
            model, return_layers={layers_to_return[-1]: "out"}
        )
        # print(f"Returning layers before {pooling_layer_name}")

    return model, last_conv_out_channels


class CustomFCN(nn.Module):
    def __init__(self, backbone, num_classes):
        super(CustomFCN, self).__init__()
        # return_layers = {"layer4": "out"}
        # self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.backbone, out_channels = replace_pool_and_get_layers(backbone)
        self.classifier = FCNHead(out_channels, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        with torch.no_grad():
            features = self.backbone(x)
            x = features["out"]
        x = self.classifier(x)
        x = nn.functional.interpolate(
            x, size=input_shape, mode="bilinear", align_corners=False
        )
        return x


def transform(image):
    image = F.resize(image, (224, 224))
    image = F.to_tensor(image)
    image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image


def target_transform(mask):
    mask = F.resize(
        mask, (224, 224), interpolation=transforms.InterpolationMode.NEAREST
    )
    array_mask = np.array(mask)
    array_mask[array_mask == 255] = 20
    mask = torch.as_tensor(array_mask, dtype=torch.long)
    return mask


def validate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.cuda()
            targets = targets.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += targets.nelement()
            correct += (predicted == targets).sum().item()
    return correct / total


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total = 0
    correct = 0
    running_loss = 0.0
    for images, targets in dataloader:
        images = images.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            total += targets.nelement()
            correct += (predicted == targets).sum().item()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return correct / total, epoch_loss


def segment_eval(args):
    args = get_segment_args(args)

    model_path = os.path.join(args.save_directory, "model.pt")
    backbone = torch.load(model_path, map_location="cpu")
    logging = Logger(args.save_directory, "eval-segment-log")

    logging("Evaluation over VOC2012")

    train_dataset = VOCSegmentation(
        root="./linslab/data",
        year="2012",
        image_set="train",
        download=False,
        transform=transform,
        target_transform=target_transform,
    )
    val_dataset = VOCSegmentation(
        root="./linslab/data",
        year="2012",
        image_set="val",
        download=False,
        transform=transform,
        target_transform=target_transform,
    )

    model = CustomFCN(backbone=backbone, num_classes=21)

    for param in model.backbone.parameters():
        param.requires_grad = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.segment_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.segment_batch_size,
        drop_last=False,
        num_workers=args.num_workers,
    )

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    model.cuda()

    logging(str_with_dashes("Eval start"))
    valid_acc = validate(model, val_loader)
    logging(
        f"epoch: [{str(0).zfill(4)}], train loss: ({0:.4f}), valid accuracy: ({valid_acc:.4f})"
    )
    accs = []
    for epoch in range(args.segment_epochs):
        train_acc, train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        valid_acc = validate(model, val_loader)
        accs.append(valid_acc * 100)
        logging(
            f"epoch: [{str(epoch+1).zfill(4)}], train accuracy: ({train_acc:.4f}), valid accuracy: ({valid_acc:.4f})"
        )

    result = f"{statistics.mean(accs[-3:]):.1f} $\pm$ {statistics.stdev(accs[-3:]):.1f}"

    logging(f"Final accuracy: {result}")

    logging(str_with_dashes(""))


# if __name__ == "__main__":

#     class a:
#         pass

#     args = a()
#     args.save_directory = (
#         "./outputs/models/byol_128_224_1_CIFAR10_resnet18_0.001_0.01/00002"
#     )
#     args.segment_epochs = 5
#     args.segment_batch_size = 8
#     segment_eval(args)
