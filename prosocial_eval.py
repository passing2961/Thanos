import os
import json
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict

from safety_model.canary import Canary


canary = Canary()

def load_results(args):
    path = os.path.join('outputs', args.task_type, args.dataset_name, 'seed:0/debug', f'{args.task_type}_{args.model_name}_{args.dataset_name}_results.json')
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", default=None, type=str)
    parser.add_argument("--dataset-name", default=None, type=str)
    parser.add_argument("--model-name", default=None, type=str)
    args = parser.parse_args()

    results = load_results(args)[:10]
    safety_ratio = defaultdict(int)

    for result in tqdm(results):
        resp = result['model_response'].strip()

        safety_result = canary.chirp(resp)

        if 'needs_caution' in safety_result:
            safety_ratio['caution'] += 1
        elif 'needs_intervention' in safety_result:
            safety_ratio['intervention'] += 1
        elif 'casual' in safety_result:
            safety_ratio['casual'] += 1
        else:
            assert False

    report_save_dir = f'reports/prosocial/{args.task_type}'
    os.makedirs(report_save_dir, exist_ok=True)

    with open(os.path.join(report_save_dir, f'{args.model_name}_{args.dataset_name}_scores.json'), 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent='\t')

