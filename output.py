from typing import Callable, Optional
import os
import json
import copy
from pathlib import Path

from collections import defaultdict
import pandas as pd
from accelerate import Accelerator

from utils.constant import *
from utils.common import *


class PredictionHandler(object):
    def __init__(
        self,
        output_dir=EVAL_OUTPUT_ROOT,
        inputs=None,
        generations=None,
        task_type=None
    ):
        self.output_dir = Path(output_dir)
        self.inputs = inputs
        self.generations = generations
        self.task_type = task_type

    def evaluate_response(
        self, 
        model_name: str, 
        benchmark_name: str, 
        accel: Accelerator,
        benchmark_eval_fn: Callable,
    ):
        # gathering all gpu to one device
        self.inputs = accel.gather_for_metrics(self.inputs)
        self.generations = accel.gather_for_metrics(self.generations)
        #self.save_feat = accel.gather_for_metrics(self.save_feat)

        if accel.is_main_process:
            # check for duplicates
            self.inputs, self.generations = remove_duplicate(benchmark_name, self.inputs, self.generations)
            
            if benchmark_name in "empathy":
                return self.evaluate_empathy(model_name, accel, benchmark_eval_fn)
            elif benchmark_name in "dailydialog":
                return self.evaluate_dailydialog(model_name, accel, benchmark_eval_fn)
            elif benchmark_name in "prosocial":
                return self.evaluate_prosocial(model_name, accel, benchmark_eval_fn)
            elif benchmark_name == "bst":
                return self.evaluate_bst(model_name, accel, benchmark_eval_fn)
            elif benchmark_name == "ours":
                return self.evaluate_ours(model_name, accel, benchmark_eval_fn)
            elif benchmark_name == "photochat":
                return self.evaluate_photochat(model_name, accel, benchmark_eval_fn)
        else:
            return None
    
    def evaluate_photochat(
        self,
        model_name: str,
        accel: Accelerator,
        benchmark_eval_fn: Callable,
    ):
        pred_answers = []
        for inputs, answer in zip(self.inputs, self.generations):
            #print(answer)
            try:
                if '[RESULT SKILL]' in answer:
                    rationale, skill = answer.split('[RESULT SKILL]')
                elif 'RESULT SKILL:' in answer:
                    rationale, skill = answer.split('RESULT SKILL:')
                else:
                    rationale = ''
                    skill = ''
            except:
                rationale = ''
                skill = ''

            cp_inputs = copy.deepcopy(inputs)
            cp_inputs['pred_skill'] = skill.strip()
            cp_inputs['pred_explanation'] = rationale.strip()
            pred_answers.append(cp_inputs)

        pred_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_photochat_results.json')
        json.dump(pred_answers, open(pred_pth, 'w'))
        accel.print(f"Finished annotating PhotoChat. The result file saved to {pred_pth}.")

        report = benchmark_eval_fn(pred_answers, task_type=self.task_type)
        score_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_photochat_scores.json')
        json.dump(report, open(score_pth, 'w'))
        accel.print(f"[PhotoChat] Performance: {report}")

    def evaluate_ours(
        self,
        model_name: str,
        accel: Accelerator,
        benchmark_eval_fn: Callable,
    ):
        pred_answers = []
        for inputs, answer in zip(self.inputs, self.generations):
            #print(answer)
            try:
                if '[RESULT SKILL]' in answer:
                    rationale, skill = answer.split('[RESULT SKILL]')
                elif 'RESULT SKILL:' in answer:
                    rationale, skill = answer.split('RESULT SKILL:')
                else:
                    rationale = ''
                    skill = ''
            except:
                rationale = ''
                skill = ''

            cp_inputs = copy.deepcopy(inputs)
            cp_inputs['pred_skill'] = skill.strip()
            cp_inputs['pred_explanation'] = rationale.strip()
            pred_answers.append(cp_inputs)

        pred_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_ours_results.json')
        json.dump(pred_answers, open(pred_pth, 'w'))
        accel.print(f"Finished annotating Ours. The result file saved to {pred_pth}.")

        report = benchmark_eval_fn(pred_answers, task_type=self.task_type)
        score_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_ours_scores.json')
        json.dump(report, open(score_pth, 'w'))
        accel.print(f"[Ours] Performance: {report}")

    def evaluate_bst(
        self,
        model_name: str,
        accel: Accelerator,
        benchmark_eval_fn: Callable,
    ):
        if 'skill' in self.task_type:
            pred_answers = []
            for inputs, answer in zip(self.inputs, self.generations):
                try:
                    if '[RESULT SKILL]' in answer:
                        parsed_result = answer.split('[RESULT SKILL]')
                        if len(parsed_result) > 2:
                            rationale, skill = parsed_result[0], parsed_result[1]
                        else:
                            rationale, skill = parsed_result
                    elif 'RESULT SKILL:' in answer:
                        rationale, skill = answer.split('RESULT SKILL:')
                    else:
                        rationale, skill = "", ""  # 기본값 설정
                    
                    # 깊은 복사
                    cp_inputs = copy.deepcopy(inputs)
                    cp_inputs['skill'] = skill.strip()
                    cp_inputs['rationale'] = rationale.strip()
                    pred_answers.append(cp_inputs)
                except Exception as e:
                    accel.print(f"Error processing answer: {answer}, Error: {str(e)}")
            
        
        pred_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_bst_results.json')
        json.dump(pred_answers, open(pred_pth, 'w'))
        accel.print(f"Finished annotating BST. The result file saved to {pred_pth}.")

        report = benchmark_eval_fn(pred_answers, task_type=self.task_type)
        score_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_bst_scores.json')
        json.dump(report, open(score_pth, 'w'))
        accel.print(f"[BST] Performance: {report}")

    def evaluate_prosocial(
        self,
        model_name: str,
        accel: Accelerator,
        benchmark_eval_fn: Callable,
    ):
        if 'next_resp' in self.task_type:
            pred_answers = []
            for inputs, answer in zip(self.inputs, self.generations):
                cp_inputs = copy.deepcopy(inputs)
                cp_inputs['model_response'] = answer.strip()
                pred_answers.append(cp_inputs)
        elif 'skill' in self.task_type:
            pred_answers = []
            for inputs, answer in zip(self.inputs, self.generations):
                try:
                    if '[RESULT SKILL]' in answer:
                        parsed_result = answer.split('[RESULT SKILL]')
                        if len(parsed_result) > 2:
                            rationale, skill = parsed_result[0], parsed_result[1]
                        else:
                            rationale, skill = parsed_result
                    elif 'RESULT SKILL:' in answer:
                        rationale, skill = answer.split('RESULT SKILL:')
                    else:
                        rationale, skill = "", ""  
                    
                    cp_inputs = copy.deepcopy(inputs)
                    cp_inputs['skill'] = skill.strip()
                    cp_inputs['rationale'] = rationale.strip()
                    pred_answers.append(cp_inputs)
                except Exception as e:
                    accel.print(f"Error processing answer: {answer}, Error: {str(e)}")
        elif 'doctor' in self.task_type:
            pred_answers = []
            for inputs, answer in zip(self.inputs, self.generations):
                cp_inputs = copy.deepcopy(inputs)
                cp_inputs['doctor'] = answer.strip()
                pred_answers.append(cp_inputs)
        

        pred_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_prosocial_results.json')
        json.dump(pred_answers, open(pred_pth, 'w'))
        accel.print(f"Finished annotating ProsocialDialogue. The result file saved to {pred_pth}.")
        
        report = benchmark_eval_fn(pred_answers, task_type=self.task_type)

        score_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_prosocial_scores.json')
        json.dump(report, open(score_pth, 'w'))
        accel.print(f"[Prosocial Dialogue] Accuracy: {report}")

    def evaluate_thanos(
        self, 
        model_name: str, 
        accel: Accelerator,
        benchmark_eval_fn: Callable,
        benchmark_name: str
    ):
        pred_answers = []
        for inputs, answer in zip(self.inputs, self.generations):
            inputs['skill'] = answer
            pred_answers.append(inputs)

        pred_pth = os.path.join(self.output_dir, f'{model_name}_{benchmark_name}.json')                
        json.dump(pred_answers, open(pred_pth, 'w'))
        
        accel.print(f"Finished annotating {benchmark_name}. The result file saved to {pred_pth}.")
        

    def evaluate_dailydialog(
        self, 
        model_name: str, 
        accel: Accelerator,
        benchmark_eval_fn: Callable
    ):

        if 'next_resp' in self.task_type:
            pred_answers = []
            for inputs, answer in zip(self.inputs, self.generations):
                cp_inputs = copy.deepcopy(inputs)
                cp_inputs['model_response'] = answer.strip()
                pred_answers.append(cp_inputs)
        elif 'skill' in self.task_type:
            pred_answers = []
            for inputs, answer in zip(self.inputs, self.generations):
                #print(answer)
                if '[RESULT SKILL]' in answer:
                    rationale, skill = answer.split('[RESULT SKILL]')
                else:
                    assert False
                cp_inputs = copy.deepcopy(inputs)
                cp_inputs['skill'] = skill.strip()
                cp_inputs['rationale'] = rationale.strip()
                pred_answers.append(cp_inputs)
        elif 'doctor' in self.task_type:
            pred_answers = []
            for inputs, answer in zip(self.inputs, self.generations):
                cp_inputs = copy.deepcopy(inputs)
                cp_inputs['doctor'] = answer.strip()
                pred_answers.append(cp_inputs)

        pred_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_dailydialog_results.json')
        json.dump(pred_answers, open(pred_pth, 'w'))
        accel.print(f"Finished annotating DailyDialog. The result file saved to {pred_pth}.")

        
        if 'next_resp' in self.task_type:
            report = benchmark_eval_fn(pred_answers)
            accel.print(f"Finished evaluating DailyDialog. Evaluate the result file saved to {pred_pth}.")
            
            score_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_dailydialog_scores.json')
            json.dump(report, open(score_pth, 'w'))
            accel.print(f"DailyDialog Performance: {report}")
        

    def evaluate_empathy(
        self, 
        model_name: str, 
        accel: Accelerator,
        benchmark_eval_fn: Callable
    ):
        
        if 'next_resp' in self.task_type:
            pred_answers = []
            for inputs, answer in zip(self.inputs, self.generations):
                cp_inputs = copy.deepcopy(inputs)
                cp_inputs['model_response'] = answer.strip()
                pred_answers.append(cp_inputs)
        elif 'skill' in self.task_type:
            pred_answers = []
            for inputs, answer in zip(self.inputs, self.generations):
                #print(answer)
                if '[RESULT SKILL]' in answer:
                    rationale, skill = answer.split('[RESULT SKILL]')
                else:
                    assert False
                cp_inputs = copy.deepcopy(inputs)
                cp_inputs['skill'] = skill.strip()
                cp_inputs['rationale'] = rationale.strip()
                pred_answers.append(cp_inputs)
        elif 'doctor' in self.task_type:
            pred_answers = []
            for inputs, answer in zip(self.inputs, self.generations):
                cp_inputs = copy.deepcopy(inputs)
                cp_inputs['doctor'] = answer.strip()
                pred_answers.append(cp_inputs)
                
        pred_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_empathy_results.json')
        json.dump(pred_answers, open(pred_pth, 'w'))
        accel.print(f"Finished evaluating Empathy. Evaluate the result file saved to {pred_pth}.")
        


        if 'next_resp' in self.task_type:
            report = benchmark_eval_fn(pred_answers)
            accel.print(f"Finished evaluating Empathy. Evaluate the result file saved to {pred_pth}.")
            
            score_pth = os.path.join(self.output_dir, f'{self.task_type}_{model_name}_empathy_scores.json')
            json.dump(report, open(score_pth, 'w'))
            accel.print(f"Empathy Performance: {report}")
        