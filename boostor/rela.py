import torch.nn as nn
import torchvision
import torch
import math
from math import ceil
import torch.nn.functional as F
import random
from copy import deepcopy
from tqdm import tqdm
from utilities import load_normalize
import os


def add_data_updator(epoch_end):
    def warpper(_self):
        epoch_end(_self)
        _self.dataset._update_re_induces()

    return warpper


def add_rand(METHOD, dataset, args):
    class add_randdata:
        def __init__(self, dataset, ratio=1):

            self.ratio = ratio
            self.indices = range(len(dataset))
            self._dataset = dataset
            self.re_indices = random.sample(
                self.indices, int(len(self.indices) * self.ratio)
            )

        def __len__(self):
            return len(self.re_indices)

        def __getitem__(self, idx):
            idx = self.re_indices[idx]
            data = self._dataset[idx]
            return data

        def _update_re_induces(self):
            self.re_indices = random.sample(
                self.indices, int(len(self.indices) * self.ratio)
            )

    dataset = add_randdata(dataset, args.data_ratio)

    class MODEL(METHOD):
        def __init__(self, *args):
            super(MODEL, self).__init__(*args)
            self.dataset = dataset

    MODEL.on_train_epoch_end = add_data_updator(METHOD.on_train_epoch_end)

    return MODEL


def add_rela(METHOD, dataset, observer, args):
    dataset = add_reladata(
        dataset,
        args.input_size,
        observer,
        args.rela_input_size,
        args.feature_dim,
        args.data_ratio,
        args.rela_data_path,
    )
    METHOD = add_relaloss(
        METHOD,
        args.feature_dim,
        dataset.Y.shape[-1],
        dataset,
        args.grad_accu,
        args.rela_conlam,
        args.epochs,
    )
    return METHOD


def rela_training_step(_self, x_i, x_j, y_):
    h0 = _self.backbone(x_i).flatten(start_dim=1)
    y_hat_i = _self.transfer(h0)
    h1 = _self.backbone(x_j).flatten(start_dim=1)
    y_hat_j = _self.transfer(h1)
    loss = 0.5 * (_self.rela_criterion(y_hat_i, y_) + _self.rela_criterion(y_hat_j, y_))
    _self.record("train loss", loss.item())
    return loss, h0.detach(), h1.detach()


def add_relaloss(
    METHOD,
    in_dim=512,
    out_dim=512,
    dataset=None,
    grad_accu=1,
    conlam=None,
    epochs=100,
):
    def add_loss(loss_func):

        def wrapper(_self, data, idx):
            # _data, rela_data = data
            rela_data = data

            if _self.rela_rho < 0.995:
                x_i, x_j, y_, y = rela_data
                loss, h0, h1 = rela_training_step(_self, x_i, x_j, y_)

                cur_loss = _self.all_gather(loss.item()).mean()
                _self.accu_loss.append(cur_loss)
                if idx % grad_accu == grad_accu - 1:
                    cur_loss = sum(_self.accu_loss) / len(_self.accu_loss)
                    _self.accu_loss = []
                    _self.rela_rho = math.exp(
                        -max(_self.rela_loss - _self.rela_loss_fast, 0) * 1
                    )
                    # _self.rela_rho = _self.all_gather(_self.rela_rho).mean()
                    _self.rela_loss_fast = (
                        _self.rela_loss_fast * 0.999 + cur_loss * 0.001
                    )
                    _self.rela_loss = (
                        _self.rela_loss * 0.99 + _self.rela_loss_fast * 0.01
                    )
                    # print(_self.rela_rho)
                    # print(_self.rela_loss)

                loss += _self.update_step(h0, h1)
                loss += _self.online_classifier_training_step(h0, y)

            elif _self.rela_rho == 1.0:
                loss = loss_func(_self, ((data[0], data[1]), data[3]), idx)

            else:
                _self.rela_rho = 1.0
                _self.record("fast learning end", _self.global_step)
                print("fast learning end")
                loss = loss_func(_self, ((data[0], data[1]), data[3]), idx)
                _self.dataset._dataset.transform = _self.dataset.oalg_transform

            return loss

        def wrapper_conslam(_self, data, idx):

            rela_data = data

            if _self.current_epoch < int(conlam * epochs):
                x_i, x_j, y_, y = rela_data
                loss, h0, h1 = rela_training_step(_self, x_i, x_j, y_)

                loss += _self.update_step(h0, h1)
                loss += _self.online_classifier_training_step(h0, y)

            elif _self.rela_rho == 1.0:
                loss = loss_func(_self, ((data[0], data[1]), data[3]), idx)

            else:
                _self.rela_rho = 1.0
                _self.record("fast learning end", _self.global_step)
                print("fast learning end")
                loss = loss_func(_self, ((data[0], data[1]), data[3]), idx)
                _self.dataset._dataset.transform = _self.dataset.oalg_transform

            return loss

        if conlam is None:
            return wrapper
        else:
            return wrapper_conslam

    class MODEL(METHOD):

        def __init__(self, *args):
            super(MODEL, self).__init__(*args)
            self.transfer = nn.Linear(in_dim, out_dim)
            self.dataset = dataset
            self.rela_loss = 2.0
            self.rela_loss_fast = 1.0
            self.rela_rho = 0.0
            self.accu_loss = []

        def rela_criterion(self, x, y):
            return (1 - nn.CosineSimilarity(dim=1)(x, y)).mean()

    MODEL.training_step = add_loss(METHOD.training_step)
    MODEL.on_train_epoch_end = add_data_updator(METHOD.on_train_epoch_end)

    return MODEL


class TransformReLA:
    def __init__(self, input_size):

        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(
                    size=input_size, scale=(0.5, 1.0)
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                load_normalize("imagenet-1k"),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class add_reladata:
    def __init__(
        self, dataset, input_size, observer, obs_input_size, repdim, ratio=1, path=None
    ):
        self.rela_transform = TransformReLA(input_size)
        self.oalg_transform = dataset.transform
        self.indices = range(len(dataset))
        self._dataset = dataset
        self.ratio = ratio

        self._update(dataset, observer, obs_input_size, repdim, path)

    def __len__(self):
        return len(self.re_indices)

    def __getitem__(self, idx):
        idx = self.re_indices[idx]
        data = self._dataset[idx]
        data = (data[0][0], data[0][1], self.Y[idx], data[1])
        return data

    def _update_re_induces(self):
        self.re_indices = random.sample(
            self.indices, int(len(self.indices) * self.ratio)
        )

    def _update(self, dataset, observer, obs_size, repdim, path):

        dataset.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=obs_size),
                torchvision.transforms.CenterCrop(obs_size),
                torchvision.transforms.ToTensor(),
                load_normalize("imagenet-1k"),
            ]
        )

        if os.path.exists(path):
            print("load saved pseudo representation targets")
            self.Y = torch.load(path, map_location=torch.device("cpu"))
        else:
            print("start generating pseudo representation targets")

            observer = deepcopy(observer)

            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=256,
                shuffle=False,
                num_workers=8,
            )

            with torch.no_grad():
                self.Y = []
                observer = observer.eval().cuda()
                for x, _ in tqdm(data_loader):
                    y = observer(x.cuda(non_blocking=True)).float().cpu()
                    self.Y.append(y)

                self.Y = torch.cat(self.Y, dim=0)

                torch.save(self.Y, path)

        with torch.no_grad():
            self.Y = self._pca_reduce(self.Y, min(repdim, self.Y.shape[-1])).cpu()
            print(f"representation targets dimension: {self.Y.shape}")

        self._dataset.transform = self.rela_transform
        self._update_re_induces()

    def _pca_reduce(self, Y, n_components):
        n_samples = Y.shape[0]
        batch_size = min(50000, n_samples)
        Y_mean = Y.mean(dim=0)
        Y_centered = Y - Y_mean

        # Initialize covariance matrix
        covariance_matrix = torch.zeros(
            (Y.shape[1], Y.shape[1]), dtype=torch.float32, device="cuda"
        )

        # Batch processing for covariance matrix
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = Y_centered[start:end].cuda()
            covariance_matrix += torch.mm(batch.t(), batch)

        covariance_matrix /= n_samples - 1

        # Compute eigenvectors
        _, eigenvectors = torch.linalg.eigh(covariance_matrix, UPLO="U")
        principal_components = eigenvectors[:, -n_components:]

        # Batch processing for reduced data
        Y_reduced = torch.zeros(
            (n_samples, n_components), dtype=torch.float32, device="cpu"
        )
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = Y_centered[start:end].cuda()
            Y_reduced[start:end] = torch.mm(batch, principal_components)

        # Normalize
        Y_reduced = self._normalize_features(Y_reduced)

        return Y_reduced

    def _normalize_features(self, Y):
        n_samples = Y.shape[0]
        batch_size = min(50000, n_samples)
        dim = Y.shape[1]
        global_sum = torch.zeros(dim, dtype=torch.float32, device="cuda")
        global_sqr_sum = torch.zeros(dim, dtype=torch.float32, device="cuda")

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = Y[start:end].cuda()
            global_sum += batch.sum(dim=0)
            global_sqr_sum += (batch**2).sum(dim=0)

        global_mean = global_sum / n_samples
        global_var = (global_sqr_sum / n_samples) - (global_mean**2)
        global_std = torch.sqrt(global_var)

        Y_normalized = (Y - global_mean.cpu()) / global_std.cpu()

        return Y_normalized
