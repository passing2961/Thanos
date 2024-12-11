from typing import Union, List
import os

from rich.progress import Progress
from rich.console import Console
from tqdm import tqdm

from benchmarks_zoo import list_benchmarks, load_benchmark
from output import PredictionHandler
from utils.common import *
from utils.constant import *


class Evaluator(object):
    def __init__(
        self, 
        config,
        seed: int = 0,
        benchmarks: Union[List[str], str] = "all"
    ):
        self.config = config

        self.update_benchmark_list(benchmarks)

    def update_benchmark_list(self, benchmarks: Union[List[str], str]):
        if isinstance(benchmarks, str):
            self.benchmarks = list_benchmarks(benchmarks)
        elif isinstance(benchmarks, list):
            self.benchmarks = benchmarks
        
        assert (
            isinstance(self.benchmarks, list) and len(self.benchmarks) > 0
        ), "Please provide benchmarks to evaluate!"
        Console().print("There are {} benchmarks to evaluate".format(len(self.benchmarks)))

    def reset(self):
        self.inputs = []
        self.generations = []
    
    def add(self, inputs, outputs):
        self.inputs.extend(inputs)
        self.generations.extend(outputs)

    def evaluate_response(self, model, accel):

        results = {}
        with Progress(transient=True) as progress:
            pg_benchmarks = progress.add_task(
                "[green]Processing...", total=len(self.benchmarks)
            )

            for benchmark_name in self.benchmarks:
                progress.update(
                    pg_benchmarks,
                    description=f"[green]Processing {benchmark_name}..."
                )

                output_dir = os.path.join(
                    EVAL_OUTPUT_ROOT, 
                    self.config.task_type,
                    benchmark_name, 
                    f'seed:{self.config.seed}'
                )
                if self.config.debug:
                    output_dir = os.path.join(output_dir, 'debug')

                if 'thanos_next_resp' in self.config.task_type:
                    output_dir = os.path.join(output_dir, self.config.thanos_model_size)
                    
                os.makedirs(output_dir, exist_ok=True)

                eval_benchmark = load_benchmark(benchmark_name, self.config)
                
                self.reset()
                eval_dataloader = torch.utils.data.DataLoader(
                    eval_benchmark,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=16,
                    pin_memory=True,
                    collate_fn=lambda x: x
                )

                # Accel distributed
                eval_dataloader = accel.prepare(eval_dataloader)

                # progress bar
                prog_bar = tqdm(
                    eval_dataloader, disable=not accel.is_local_main_process, total=len(eval_dataloader)
                )
                # eval start
                for inputs in prog_bar:
                    # memory opt
                    memory_optimization()

                    with torch.inference_mode():
                        generations = model.generate(
                            inputs=inputs, 
                            device=accel.device
                        )
                    
                    for item in inputs:
                        if 'image' in item:
                            del item['image']
                    
                    self.add(inputs, generations)
                
                Console().print(f"[Device: {accel.device}] Finished!")
                accel.wait_for_everyone()

                # memory opt
                memory_optimization()

                
                pred_handler = PredictionHandler(
                    output_dir=output_dir,
                    inputs=self.inputs,
                    generations=self.generations,
                    task_type=self.config.task_type
                )
                results[benchmark_name] = pred_handler.evaluate_response(
                    model_name=self.config.model_name,
                    benchmark_name=benchmark_name,
                    accel=accel,
                    benchmark_eval_fn=eval_benchmark.evaluate_response
                )
        accel.print(results)
        return