import copy
from methods.base import Base
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from lightly.models import ResNetGenerator, modules, utils
from lightly.models.modules import heads, memory_bank


class DINO(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.head = self._build_projection_head(self.args.feature_dim)
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = self._build_projection_head(self.args.feature_dim)

        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048)

    def _build_projection_head(self, feature_dim):
        head = heads.DINOProjectionHead(feature_dim, 2048, 256, 2048, batch_norm=True)
        # use only 2 layers for cifar10
        head.layers = heads.ProjectionHead(
            [
                (feature_dim, 2048, nn.BatchNorm1d(2048), nn.GELU()),
                (2048, 256, None, None),
            ]
        ).layers
        return head

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.head(y)
        return y, z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def update_step(self, h0, h1):
        utils.update_momentum(self.backbone, self.teacher_backbone, m=0.99)
        utils.update_momentum(self.head, self.teacher_head, m=0.99)
        return 0

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.teacher_backbone, m=0.99)
        utils.update_momentum(self.head, self.teacher_head, m=0.99)
        views, y = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = []
        for view in views:
            h, z = self.forward(view)
            student_out.append(z)
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.record("train loss", loss.item())

        loss_ce = self.online_classifier_training_step(h, y)
        return loss + loss_ce

    def configure_optimizers(self):
        params = (
            list(self.backbone.parameters())
            + list(self.head.parameters())
            + list(self.classifier.parameters())
        )

        return torch.optim.AdamW(
            params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )


def TransformDINO(dataset, input_size):

    if dataset == "CIFAR10" or dataset == "CIFAR100":
        transform = DINOTransform(
            global_crop_size=input_size,
            n_local_views=0,
            cj_strength=0.5,
            gaussian_blur=(0, 0, 0),
        )
    else:
        transform = DINOTransform(
            global_crop_size=input_size,
            n_local_views=0,
        )
    return transform
