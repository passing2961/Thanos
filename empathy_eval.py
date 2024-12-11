import os
import json
import argparse
from tqdm import tqdm

from evaluation import _metrics


def load_results(args):
    path = os.path.join('outputs', args.task_type, args.dataset_name, 'seed:0/debug', f'{args.task_type}_{args.model_name}_{args.dataset_name}_results.json')
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", default=None, type=str)
    parser.add_argument("--dataset-name", default=None, type=str)
    parser.add_argument("--model-name", default=None, type=str)
    parser.add_argument(
        "--empintent_ckpt", 
        type=str, 
        default="evaluation/models/empintent/bert-base-uncased"
    )
    parser.add_argument(
        "--emotion_ckpt", 
        type=str, 
        default="evaluation/models/emotion/bert-base-uncased"
    )
    parser.add_argument(
        "--epitome_ckpt", 
        type=str, 
        default="evaluation/models/epitome"
    )

    args = parser.parse_args()

    results = load_results(args)
    all_pred_resp = []
    for result in tqdm(results, total=len(results)):
        all_pred_resp.append(result['model_response'].strip().lower())

    metrics = _metrics
    print("Metrics to be used in our evaluation: {}".format(metrics.keys())) 

    report = {}
    for name, metric in metrics.items():
        if name == 'empintent':
            _, value = metric(args.empintent_ckpt, args.model_name).calculate(results)
        elif name == 'emotion':
            _, value = metric(args.emotion_ckpt, args.model_name).calculate(results)
        elif name == 'epitome':
            value = metric(args.epitome_ckpt, args.model_name).calculate(results)
        else:
            value = metric().calculate(all_pred_resp)

        report[name] = value

    report_save_dir = f'reports/empathy/{args.task_type}'
    os.makedirs(report_save_dir, exist_ok=True)

    with open(os.path.join(report_save_dir, f'{args.model_name}_{args.dataset_name}_scores.json'), 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent='\t')

