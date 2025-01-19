import copy
import torch
import torchvision
from torch import nn
from methods.base import Base
import pytorch_lightning as pl
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.models.utils import update_momentum
from lightly.transforms import BYOLTransform


class BYOL(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.projection_head = BYOLProjectionHead(
            self.args.feature_dim, self.args.feature_dim * 2, 256
        )
        self.prediction_head = BYOLPredictionHead(256, self.args.feature_dim * 2, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p, y

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def update_step(self, h0, h1):
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
        return 0

    def training_step(self, batch, batch_idx):
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
        (x0, x1), y = batch
        p0, _ = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1, h = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.record("train loss", loss.item())

        loss_ce = self.online_classifier_training_step(h, y)
        return loss + loss_ce

    def configure_optimizers(self):
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
            + list(self.classifier.parameters())
        )

        return torch.optim.AdamW(
            params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )


def TransformBYOL(dataset, input_size):
    if dataset == "CIFAR10" or dataset == "CIFAR100":
        transform = BYOLTransform(
            view_1_transform=BYOLView1Transform(
                input_size=input_size, gaussian_blur=0.0
            ),
            view_2_transform=BYOLView2Transform(
                input_size=input_size, gaussian_blur=0.0
            ),
        )
    else:
        transform = BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=input_size),
            view_2_transform=BYOLView2Transform(input_size=input_size),
        )
    return transform
