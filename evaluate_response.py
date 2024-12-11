import os
import argparse

from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from rich.console import Console

from evaluator import Evaluator
from utils.common import *
from models_zoo.registry import load_model



def evaluate(args):
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Accelerator for DDP, FSDP, DeepSpeed, etc [Should First Call]
    accel = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])

    if accel.is_main_process:
        Console().print(args)
    
    # loading model
    model = load_model(args, accel)

    # loading evaluator
    evaluator = Evaluator(args, benchmarks=args.dataset.split(','))
    
    # Update dataset and evaluator
    evaluator.evaluate_response(model, accel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--model-path', default=None, type=str)
    parser.add_argument('--model-name', default=None, type=str)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--max-new-tokens', default=512, type=int)
    parser.add_argument('--top-p', default=0., type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cache-dir', default=None, type=str)
    
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--task-type', default=None, type=str, help='skill|resp_gen')
    parser.add_argument('--use-skill-annot', action='store_true', default=False)
    parser.add_argument('--for-human-eval', action='store_true', default=False)
    parser.add_argument('--thanos-model-size', default=None, type=str)
    args = parser.parse_args()

    seed_everything(args.seed)

    evaluate(args)

