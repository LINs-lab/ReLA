import os
import json
import time
import torch
from torch import nn
from copy import copy
import pytorch_lightning as pl
from utilities import Logger, get_max_memory, str_with_dashes, Recorder


class Base(pl.LightningModule):
    def __init__(self, backbone, args):
        super().__init__()

        args.save_directory = os.path.join(args.root_directory, args.seed)
        self.classifier = nn.Linear(args.feature_dim, args.nclass)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.backbone = backbone
        self.args = args

        self.global_epoch = 0
        self.valid_step_outputs = []

        self.record = Recorder(args.save_directory)
        self.record.dict = {
            "epoch index": [0],
            "train time": [0],
            "epoch time": [0],
        }
        # self.record.dict = {
        #     "epoch index": [0],
        #     "train loss": [],
        #     "valid loss": [],
        #     "train accuracy": [],
        #     "valid accuracy": [],
        #     "train time": [0],
        #     "epoch time": [0],
        # }

    def report_performance(self):
        if self.valid_step_outputs != []:
            all_preds = torch.stack(self.valid_step_outputs).mean(0)
            self.record("valid accuracy", all_preds[0].item())
            self.record("valid loss", all_preds[1].item())

        if self.trainer.is_global_zero:
            self.logging(
                f"epoch: [{str(self.global_epoch).zfill(4)}], valid accuracy: ({self.record.dict['valid accuracy'][-1]:.4f}), train time: ({self.record.dict['train time'][-1]:.4f}), epoch time: ({self.record.dict['epoch time'][-1]:.4f})"
            )
        self.valid_step_outputs.clear()

    def online_classifier_training_step(self, h, y):
        y_hat = self.classifier(h.detach())
        loss_ce = self.ce_criterion(y_hat, y)
        with torch.no_grad():
            acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
            self.record("train accuracy", acc.item())
        return loss_ce

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if len(self.record.dict["epoch time"]) <= self.global_epoch:
            self.record("epoch time", 0)
        self.record.dict["epoch time"][self.global_epoch] = (
            time.time() - self.epoch_start_time
        )
        self.report_performance()
        self.record.dict["epoch index"].append(self.global_epoch)
        self.global_epoch += 1

    def on_train_batch_start(self, batch, batch_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_time = time.time() - self.batch_start_time
        if len(self.record.dict["train time"]) <= self.global_epoch:
            self.record("train time", 0)
        self.record.dict["train time"][self.global_epoch] += batch_time

    def validation_step(self, batch, batch_idx):
        x, y = batch
        h_hat = self.backbone(x)
        y_hat = self.classifier(h_hat)
        loss = self.ce_criterion(y_hat, y).mean()
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.valid_step_outputs.append(torch.tensor([acc, loss]))
        return loss

    def on_validation_epoch_end(self):
        if self.global_epoch == 0:
            self.report_performance()
            self.global_epoch += 1

    def on_fit_start(self):
        if self.trainer.is_global_zero:
            self.logging = Logger(self.args.save_directory, "fit-log")
            params_dict = copy(self.args.__dict__)
            del params_dict["root_directory"]
            self.logging(str_with_dashes("Parameters"))
            self.logging(json.dumps(params_dict, indent=2))
            self.logging(str_with_dashes("Training start"))

    def on_train_end(self):
        if self.trainer.num_devices > 1:
            for key in self.record.dict.keys():
                gathered_results = torch.stack(self.all_gather(self.record.dict[key]))
                if self.trainer.is_global_zero:
                    if gathered_results.dtype is torch.float32:
                        self.record.dict[key] = gathered_results.mean(1).cpu().tolist()
                    else:
                        self.record.dict[key] = gathered_results[:, 0].cpu().tolist()

        if self.trainer.is_global_zero:
            self.logging(str_with_dashes("Cost report"))
            self.logging(
                f"Total train cost: {sum(self.record.dict['train time']):.4f} seconds"
            )
            self.logging(
                f"Total cost: {sum(self.record.dict['epoch time']):.4f} seconds"
            )
            self.logging(f"Peak memory usage: {get_max_memory():.4f} GB")
            self.logging(str_with_dashes(""))
            self.record.save()
            torch.save(self.backbone, f"{self.args.save_directory}/model.pt")
