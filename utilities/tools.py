import os
import ast
import time
import torch
import random
import argparse
import numpy as np
import pytorch_lightning as pl

BASE62_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class Recorder:
    def __init__(self, save_directory=None, name="fit-log"):
        self.dict = {}
        if save_directory is not None:
            self.save_directory = save_directory
            self.save_path = os.path.join(save_directory, name + ".pth")

    def __call__(self, key, value):
        if key in self.dict:
            self.dict[key].append(value)
        else:
            self.dict[key] = [value]

    def save(self):
        # Create the directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        torch.save(self.dict, self.save_path)


def str_with_dashes(input_str: str, total_length: int = 100) -> str:
    # Calculate the number of '-' to add on each side
    side_length = (total_length - len(input_str)) // 2
    # Construct the resulting string
    result = "-" * side_length + input_str + "-" * side_length
    # If the total length is odd and the input_str length is even, or vice versa, add an extra '-'
    if len(result) < total_length:
        result += "-"
    return result


def merge_logs(args):
    log_files = [
        f"{args.save_directory}/fit-log-{rank}.txt" for rank in range(len(args.devices))
    ]
    with open(f"{args.save_directory}/fit-log.txt", "a") as outfile:
        for fname in log_files:
            if os.path.exists(fname):
                with open(fname) as infile:
                    outfile.write(infile.read())
                infile.close()
                os.remove(fname)


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def get_max_memory():
    memory_allocated_bytes = torch.cuda.max_memory_allocated(device="cuda")
    memory_allocated_gb = memory_allocated_bytes / (1024**3)
    return memory_allocated_gb


class Logger:
    def __init__(self, save_directory, name):
        os.makedirs(save_directory, exist_ok=True)
        self.logger = open(os.path.join(save_directory, name + ".txt"), "w")

    def __call__(self, string, end="\n", print_=True):
        if print_:
            print("{}".format(string), end=end)
            if end == "\n":
                self.logger.write("{}\n".format(string))
            else:
                self.logger.write("{} ".format(string))
            self.logger.flush()


def parse_list(option_str):
    try:
        option_str = option_str.strip()
        if option_str.startswith("[") and option_str.endswith("]"):
            elements = option_str[1:-1].split(",")
            processed_elements = []
            for element in elements:
                element = element.strip()
                if element.replace(".", "", 1).isdigit():
                    processed_elements.append(element)
                else:
                    processed_elements.append(f'"{element}"')
            option_str = "[" + ", ".join(processed_elements) + "]"

        result = ast.literal_eval(option_str)
        if not isinstance(result, list):
            raise ValueError
        return result
    except:
        raise argparse.ArgumentTypeError("Invalid list format")


def custom_base62_to_int(s):
    char_to_value = {char: idx for idx, char in enumerate(BASE62_CHARS)}
    base = len(BASE62_CHARS)
    num = sum(char_to_value[char] * (base**idx) for idx, char in enumerate(reversed(s)))
    return num


def set_seed(seed=None):
    seed_int = custom_base62_to_int(seed)

    random.seed(seed_int)
    os.environ["PYTHONHASHSEED"] = str(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    torch.cuda.manual_seed(seed_int)
    torch.cuda.manual_seed_all(seed_int)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.distributed.is_initialized():
        torch.distributed.manual_seed(seed_int)

    # pl.seed_everything(seed_int)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    return seed


def get_lastdir(directory, length=5):
    subdirs = [
        d
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
        and len(d) == length
        and d.isdigit()
    ]
    subdirs.sort()
    last_dir = subdirs[-1]
    return last_dir


def gen_seed(path, length=5, randomly=True):
    if randomly:
        seed = "".join(random.choice(BASE62_CHARS) for _ in range(length))
    else:
        # Check if the path exists and has subdirectories
        if not os.path.exists(path) or not os.listdir(path):
            seed = "00000"
        else:
            # Get all subdirectory names that match the expected length
            last_dir = get_lastdir(path, length)
            # If there are no subdirectories, start with "00000"
            if not last_dir:
                seed = "00000"
            else:
                # Sort subdirectory names and generate the next seed
                next_seed = int(last_dir) + 1
                seed = str(next_seed).zfill(length)
    return seed
