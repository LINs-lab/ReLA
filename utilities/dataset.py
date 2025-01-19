import os
import random
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Ignore warnings caused by the specific Torch version
import warnings

warnings.filterwarnings("ignore")


class GeneralTransform:
    def __init__(self, input_size):

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(input_size),
                torchvision.transforms.CenterCrop(input_size),
                torchvision.transforms.ToTensor(),
                load_normalize("imagenet-1k"),
            ]
        )

    def __call__(self, x):
        return self.transform(x)


class ImageFolder(torchvision.datasets.ImageFolder):

    def __init__(
        self, classes, ipc, mem=False, shuffle=False, shuffle_in_class=False, **kwargs
    ):
        """
        Custom ImageFolder class with additional features.

        Parameters:
        - classes (list): List of selected class labels.
        - ipc (int): Number of images to sample per class.
        - mem (bool, optional): If True, load images into memory. Default is False.
        - shuffle (bool, optional): If True, shuffle images within each class. Default is False.
        - **kwargs: Additional arguments to pass to the base class constructor (torchvision.datasets.ImageFolder).

        Attributes:
        - mem (bool): Flag indicating whether to load images into memory.
        - image_paths (list): List of file paths for all sampled images.
        - samples (list): List of loaded image samples (if mem=True).
        - targets (list): List of target labels for each image.

        Note: Inherits from torchvision.datasets.ImageFolder.

        """
        super(ImageFolder, self).__init__(**kwargs)
        self.mem = mem
        self.image_paths = []  # List to store file paths for all sampled images
        self.samples = []  # List to store loaded image samples (if mem=True)
        self.targets = []  # List to store target labels for each image
        class_ls = sorted(os.listdir(self.root))

        if classes == []:
            classes = range(len(class_ls))
        self.nclass = len(classes)

        # Iterate through each class
        for c in range(self.nclass):
            dir_path = os.path.join(self.root, class_ls[classes[c]])
            file_ls = sorted(os.listdir(dir_path))

            # Shuffle the file list of a class if specified
            if shuffle_in_class:
                random.shuffle(file_ls)

            if ipc == -1:
                num_samples = len(file_ls)
            else:
                num_samples = ipc

            # Sample ipc images from the class
            for i in range(num_samples):
                if i >= len(file_ls):
                    index = i - len(file_ls) * (i // len(file_ls))
                else:
                    index = i

                # Construct the full path to the image
                self.image_paths.append(os.path.join(dir_path, file_ls[index]))

                # Load the image into memory if specified
                if self.mem:
                    self.samples.append(
                        self.loader(os.path.join(dir_path, file_ls[index]))
                    )

                # Record the target label
                self.targets.append(c)

        if shuffle:
            temp = list(zip(self.image_paths, self.targets))
            random.shuffle(temp)
            self.image_paths, self.targets = zip(*temp)
            self.image_paths = list(self.image_paths)
            self.targets = list(self.targets)

    def __getitem__(self, index):
        """
        Custom implementation of __getitem__ method.

        Parameters:
        - index (int): Index of the desired item.

        Returns:
        - sample (torch.Tensor): Transformed image sample.
        - target (int): Target label for the image.
        """
        # Load the image from memory or file based on the mem attribute
        if self.mem:
            sample = self.samples[index]
        else:
            sample = self.loader(self.image_paths[index])

        # Apply transformations to the image sample
        sample = self.transform(sample)
        return sample, self.targets[index]

    def __len__(self):
        """
        Custom implementation of __len__ method.

        Returns:
        - int: Number of items in the dataset.
        """
        return len(self.targets)


def load_dataset(
    dataset="imagenet-10",
    train=True,
    transform=None,
    shuffle=False,
    shuffle_in_class=False,
    root="./linslab/data",
    ipc=-1,
    classes=[],
):
    """
    Get the specified dataset for training or validation.

    Parameters:
    - dataset (str): Name of the dataset, e.g., "imagenet-10" or "imagenet-1k".
    - ipc (int): Loaded images per class.
    - classes (list): List of class names to include in the dataset.
    - root (str): Root directory where the dataset is located.
    - train (bool): If True, return the training set; otherwise, return the validation set.
    - transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.

    Returns:
    - dataset: The requested dataset, either as an ImageFolder or a custom LMDB dataset.

    Note:
    - For "imagenet-1k-lmdb", the LMDB dataset is loaded and a subset is extracted based on specified classes and ipc.
    - For other datasets, ImageFolder is used to load the dataset from the specified directory.
    """

    # Converts the dataset to lowercase.
    dataset = dataset.lower()

    # Construct the directory path based on the dataset and split (train or val)
    if train:
        dataset_dir = os.path.join(root, dataset, "train")
    else:
        dataset_dir = os.path.join(root, dataset, "val")

    if transform == None:
        resize = transforms.Compose([])
        if dataset != "tinyimagenet" and "imagenet" in dataset:
            resize = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224)]
            )
        transform = transforms.Compose(
            [resize, transforms.ToTensor(), load_normalize(dataset)]
        )

    # Handle the case of "imagenet-1k-lmdb" separately

    dataset = ImageFolder(
        root=dataset_dir,
        classes=classes,
        ipc=ipc,
        mem=False,
        shuffle=shuffle,
        shuffle_in_class=shuffle_in_class,
        transform=transform,
    )

    if train is True:
        print("Training data sample number:", len(dataset))
    else:
        print("Test data sample number:", len(dataset))

    return dataset


def load_normalize(dataset="imagenet"):
    data_stats = {
        "cifar": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
        "imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "svhn": {"mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]},
        "mnist": {"mean": [0.1307], "std": [0.3081]},
        "fashionmnist": {"mean": [0.2861], "std": [0.3530]},
    }

    if "imagenet" in dataset:
        dataset = "imagenet"
    elif "cifar" in dataset:
        dataset = "cifar"

    return transforms.Normalize(
        mean=data_stats[dataset]["mean"], std=data_stats[dataset]["std"]
    )


def load_denormalize(dataset="imagenet"):
    data_stats = {
        "cifar": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
        "imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "svhn": {"mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]},
        "mnist": {"mean": [0.1307], "std": [0.3081]},
        "fashionmnist": {"mean": [0.2861], "std": [0.3530]},
    }

    if "imagenet" in dataset:
        dataset = "imagenet"
    elif "cifar" in dataset:
        dataset = "cifar"

    denormalize = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[
                    1 / data_stats[dataset]["std"][0],
                    1 / data_stats[dataset]["std"][1],
                    1 / data_stats[dataset]["std"][2],
                ],
            ),
            transforms.Normalize(
                mean=[
                    -data_stats[dataset]["mean"][0],
                    -data_stats[dataset]["mean"][1],
                    -data_stats[dataset]["mean"][2],
                ],
                std=[1.0, 1.0, 1.0],
            ),
        ]
    )

    return denormalize
