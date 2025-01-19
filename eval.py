from metrics import EVALUATORS
from main import get_parser, get_args
from utilities import get_lastdir
import os

if __name__ == "__main__":
    args = get_args((get_parser()))
    args.seed = get_lastdir(args.root_directory)
    args.save_directory = os.path.join(args.root_directory, args.seed)
    for eval_task in args.eval_tasks:
        evaluator = EVALUATORS[eval_task]
        evaluator(args)
