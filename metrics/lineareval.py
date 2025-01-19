import os
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import statistics
from utilities import load_dataset, load_normalize, get_lastdir, str_with_dashes, Logger


def get_linear_args(args):
    if args.seed is None:
        args.seed = get_lastdir(args.root_directory)
    return args


class FeatureData(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.eval()
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # self.backbone.eval()
        self.train_feature_vector = []
        self.train_labels_vector = []
        self.test_feature_vector = []
        self.test_labels_vector = []

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            h = self.backbone(x)
            self.train_feature_vector.append(h.float().cpu().detach())
            self.train_labels_vector.append(y.cpu())

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            h = self.backbone(x)
            self.test_feature_vector.append(h.float().cpu().detach())
            self.test_labels_vector.append(y.cpu())

    def on_train_end(self):
        # Convert lists to tensors
        train_feature_tensor = torch.cat([v for v in self.train_feature_vector], dim=0)
        train_labels_tensor = torch.cat([v for v in self.train_labels_vector], dim=0)
        test_feature_tensor = torch.cat([v for v in self.test_feature_vector], dim=0)
        test_labels_tensor = torch.cat([v for v in self.test_labels_vector], dim=0)

        # Use all_gather to collect tensors from all processes
        self.train_feature_vector = self.all_gather(train_feature_tensor).view(
            -1, train_feature_tensor.shape[-1]
        )
        self.train_labels_vector = self.all_gather(train_labels_tensor).view(-1)
        self.test_feature_vector = self.all_gather(test_feature_tensor).view(
            -1, test_feature_tensor.shape[-1]
        )
        self.test_labels_vector = self.all_gather(test_labels_tensor).view(-1)

        # Convert tensors back to numpy if necessary

        self.train_feature_vector = self.train_feature_vector.cpu().numpy()
        self.train_labels_vector = self.train_labels_vector.cpu().numpy()
        self.test_feature_vector = self.test_feature_vector.cpu().numpy()
        self.test_labels_vector = self.test_labels_vector.cpu().numpy()

    def configure_optimizers(self):
        return None


def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to("cuda:0")
        y = y.to("cuda:0")

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def test(args, loader, model, criterion):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to("cuda:0")
        y = y.to("cuda:0")

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


# class LogisticRegressor(pl.LightningModule):
#     def __init__(self, input_dim, num_classes):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, num_classes)
#         self.validation_step_outputs = []

#     def forward(self, x):
#         return self.linear(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.cross_entropy(logits, y)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.cross_entropy(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         acc = (preds == y).float().mean()
#         self.log("val_loss", loss)
#         self.log("val_accuracy", acc)
#         output = {"loss": loss, "accuracy": acc}
#         self.validation_step_outputs.append(output)
#         return output

#     def on_validation_epoch_end(self):
#         avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
#         avg_acc = torch.stack(
#             [x["accuracy"] for x in self.validation_step_outputs]
#         ).mean()
#         self.log("avg_val_loss", avg_loss)
#         self.log("avg_val_accuracy", avg_acc)
#         # print("avg_val_accuracy", avg_acc)
#         self.validation_step_outputs.clear()

#     def on_train_end(self):
#         # print(
#         #     f"Training completed. Final validation loss: {self.trainer.callback_metrics['avg_val_loss']}"
#         # )
#         if self.trainer.is_global_zero:
#             print(
#                 f"Training completed. Final validation accuracy: {self.trainer.callback_metrics['avg_val_accuracy']}"
#             )

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
#         return optimizer


def linear_eval(args):
    args = get_linear_args(args)
    model_path = os.path.join(args.save_directory, "model.pt")
    backbone_model = torch.load(model_path, map_location="cpu")
    logging = Logger(args.save_directory, "eval-linear-log")

    if args.dataset.startswith("ImageNet"):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                load_normalize("imagenet-1k"),
            ]
        )
    else:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=args.input_size),
                torchvision.transforms.ToTensor(),
                load_normalize("imagenet-1k"),
            ]
        )
    logging(f"Evaluation over {args.dataset}")
    train_dataset = load_dataset(
        dataset=args.dataset,
        train=True,
        transform=transform,
    )
    test_dataset = load_dataset(
        dataset=args.dataset,
        train=False,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    feature_data_module = FeatureData(backbone=backbone_model)

    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        accelerator="auto",
        devices=args.devices,
        strategy="ddp",
    )

    trainer.fit(
        feature_data_module, train_dataloaders=train_loader, val_dataloaders=test_loader
    )

    torch.distributed.barrier()

    backbone_model = backbone_model.cpu()

    feature_data_module = feature_data_module.cpu()

    torch.cuda.empty_cache()

    if trainer.is_global_zero:
        train_X = feature_data_module.train_feature_vector
        train_y = feature_data_module.train_labels_vector
        test_X = feature_data_module.test_feature_vector
        test_y = feature_data_module.test_labels_vector

        _train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_X), torch.from_numpy(train_y)
        )
        train_loader = torch.utils.data.DataLoader(
            _train_dataset,
            batch_size=args.logistic_batch_size,
            shuffle=True,
            # pin_memory=True,
        )

        _test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_X), torch.from_numpy(test_y)
        )
        test_loader = torch.utils.data.DataLoader(
            _test_dataset,
            batch_size=args.logistic_batch_size,
            shuffle=False,
            # pin_memory=True,
        )

        # ## Logistic Regression
        # logistic_regression_model = LogisticRegressor(
        #     train_X.shape[1], train_dataset.nclass
        # )

        # # args.logistic_epochs = 10

        # trainer = pl.Trainer(
        #     max_epochs=args.logistic_epochs,
        #     logger=False,
        #     enable_checkpointing=False,
        #     accelerator="auto",
        #     devices=1,
        #     # strategy="ddp",
        #     check_val_every_n_epoch=args.logistic_epochs,
        # )

        # trainer.fit(
        #     logistic_regression_model,
        #     train_dataloaders=train_loader,
        #     val_dataloaders=test_loader,
        # )

        logging(str_with_dashes("Eval start"))
        linear_model = nn.Linear(train_X.shape[1], train_dataset.nclass)
        linear_model = linear_model.to("cuda:0")

        optimizer = torch.optim.Adam(linear_model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()

        accs = []

        for epoch in range(args.logistic_epochs):
            loss_epoch, train_acc = train(
                args, train_loader, linear_model, criterion, optimizer
            )
            train_acc = train_acc / len(train_loader)
            with torch.no_grad():
                # final testing
                loss_epoch, val_acc = test(args, test_loader, linear_model, criterion)
                val_acc = val_acc / len(test_loader)
                accs.append(100 * val_acc)
                logging(
                    f"epoch: [{str(epoch).zfill(4)}], train accuracy: ({train_acc:.4f}), val accuracy: ({val_acc:.4f})"
                )

        result = (
            f"{statistics.mean(accs[-3:]):.1f} $\pm$ {statistics.stdev(accs[-3:]):.1f}"
        )

        logging(f"Final accuracy: {result}")

        logging(str_with_dashes(""))
