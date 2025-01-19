import os
import timm
import torch
import argparse
import torchvision
import torch.nn as nn
from methods import METHODS
import pytorch_lightning as pl
from metrics import EVALUATORS
from utilities import load_model
from utilities import load_dataset, load_normalize
from pytorch_lightning.profilers import SimpleProfiler
from utilities import parse_list, gen_seed, set_seed, merge_logs
from boostor import add_rela, get_prior_model, add_rand, prior_input_size_dict


def get_parser():
    parser = argparse.ArgumentParser(
        description="Setup for distributed and model training."
    )
    parser.add_argument(
        "--method", type=str, default=None, help="Training method to use"
    )
    parser.add_argument("--dataset", type=str, default="None", help="Dataset to use")
    parser.add_argument("--model", type=str, default="None", help="Model to use")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--seed", type=str, default=None, help="Seed for random number generators"
    )
    parser.add_argument("--input_size", type=int, default=None, help="Image size")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for training"
    )
    parser.add_argument(
        "--grad_accu", type=int, default=1, help="Accumulated steps for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay rate"
    )
    parser.add_argument(
        "--devices", type=parse_list, default=[0], help="List of device IDs"
    )
    parser.add_argument(
        "--distributed",
        type=bool,
        default=False,
        help="Use DistributedDataParallel (True)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of worker threads"
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=2, help="Number of prefetch factors"
    )
    # Task setup
    parser.add_argument(
        "--eval_tasks", type=parse_list, default=["linear"], help="Task for evaluation"
    )

    # Linear evaluation parameters
    parser.add_argument(
        "--logistic_batch_size", type=int, default=None, help="Batch size"
    )
    parser.add_argument(
        "--logistic_epochs", type=int, default=100, help="Number of epochs"
    )

    # Segmentation evaluation parameters
    parser.add_argument("--segment_batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--segment_epochs", type=int, default=10, help="Number of epochs"
    )

    # ReLA parameters
    parser.add_argument("--use_rela", type=str, default=None, help="Use ReLA or not")
    parser.add_argument(
        "--data_ratio", type=float, default=None, help="Data used ratio"
    )
    parser.add_argument(
        "--rela_conlam", type=float, default=None, help="If using constant lambda"
    )
    return parser


def get_args(parser):
    args = parser.parse_args()

    if len(args.devices) >= 2:
        args.distributed = True
    else:
        args.distributed = False

    # Usual experimental settings
    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        if args.model.endswith("modified"):
            args.input_size = 32
        else:
            args.input_size = 224
        args.logistic_batch_size = 128
        if args.batch_size is None:
            args.batch_size = 128

    elif args.dataset == "TinyImageNet":
        if args.model.endswith("modified"):
            args.input_size = 64
        else:
            args.input_size = 224
        args.logistic_batch_size = 128
        if args.batch_size is None:
            args.batch_size = 128

    elif args.dataset == "ImageNet-1K":
        args.input_size = 224
        args.logistic_batch_size = 1024
        if args.batch_size is None:
            args.batch_size = 512

    elif args.dataset == "CelebA-HQ":
        args.input_size = 224
        args.logistic_batch_size = 128
        if args.batch_size is None:
            args.batch_size = 128

    elif args.dataset == "ImageNet-21K":
        args.input_size = 224
        args.logistic_batch_size = 1024
        if args.batch_size is None:
            args.batch_size = 512

    # ReLA parameters
    if args.use_rela == "rand":
        args.rela_model = args.model
        args.rela_input_size = args.input_size
    elif args.use_rela == "none":
        pass
    elif args.use_rela is not None:
        args.rela_input_size = prior_input_size_dict[args.use_rela]

    args.rela_data_path = f"./boostor/redata/{args.dataset}_{args.use_rela}.pth"

    args.root_directory = f"./outputs/models/{parse_args(args)}"
    return args


def parse_args(args):
    params = ""
    args_dict = vars(args)
    for arg_name, arg_value in args_dict.items():
        if arg_name in [
            "method",
            "batch_size",
            "input_size",
            "epochs",
            "dataset",
            "model",
            "learning_rate",
            "weight_decay",
            "use_rela",
            "rela_ratio",
            "rela_conlam",
        ]:
            params += f"{arg_value}_"
    return params[:-1]


def main(args):
    if args.seed is None:
        args.seed = gen_seed(args.root_directory, randomly=False)

    set_seed(args.seed)

    METHOD, TRANSFORM = METHODS[args.method]

    backbone, args.feature_dim = load_model(args.model, only_backbone=True)

    transfrom = TRANSFORM(args.dataset, args.input_size)

    train_dataset = load_dataset(
        dataset=args.dataset,
        transform=transfrom,
        train=True,
    )

    if args.use_rela is not None:
        os.makedirs("./boostor/redata", exist_ok=True)
        if args.use_rela == "rand":
            METHOD = add_rela(METHOD, train_dataset, backbone, args)
        elif args.use_rela == "none":
            METHOD = add_rand(METHOD, train_dataset, args)
        else:
            rela_model = get_prior_model(args.use_rela)
            METHOD = add_rela(METHOD, train_dataset, rela_model, args)
    else:
        METHOD.dataset = train_dataset

    args.nclass = train_dataset.nclass
    model = METHOD(backbone, args)

    val_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.input_size),
            torchvision.transforms.CenterCrop(args.input_size),
            torchvision.transforms.ToTensor(),
            load_normalize("imagenet-1k"),
        ]
    )

    test_dataset = load_dataset(
        dataset=args.dataset,
        transform=val_transform,
        train=False,
        shuffle=False,
        shuffle_in_class=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=model.dataset,
        batch_size=args.batch_size // len(args.devices) // args.grad_accu,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size // len(args.devices) // args.grad_accu * 4,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    profiler = SimpleProfiler(dirpath=args.save_directory, filename="log")

    if args.distributed == False:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            devices=args.devices,
            accelerator="gpu",
            precision=16,
            logger=False,
            enable_checkpointing=False,
            accumulate_grad_batches=args.grad_accu,
            enable_progress_bar=False,
            profiler=profiler,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            devices=args.devices,
            accelerator="gpu",
            # strategy="ddp",
            strategy="ddp_find_unused_parameters_true",
            precision=16,
            sync_batchnorm=True,
            use_distributed_sampler=True,
            logger=False,
            enable_checkpointing=False,
            accumulate_grad_batches=args.grad_accu,
            enable_progress_bar=False,
            profiler=profiler,
        )

    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    if trainer.global_rank == 0:
        merge_logs(args)
    #     for eval_task in args.eval_tasks:
    #         evaluator = EVALUATORS[eval_task]
    #         evaluator(args)


if __name__ == "__main__":

    args = get_args(get_parser())
    main(args)
