from abc import abstractmethod
import os
from typing import List, Dict, Tuple
from tqdm import tqdm

import math
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


def evaluate_acc(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = (preds == labels).mean()
    return acc

class ModelCalculator:
    LABELS = NotImplementedError

    def __init__(self, model_ckpt: str, model_name: str):

        self.args = self._load_args(model_ckpt)
        self.tokenizer = self._load_tokenizer(model_ckpt)
        self.model = self._load_model(model_ckpt)
        self.model_name = model_name

    def _load_args(self, model_ckpt: str):
        return torch.load(os.path.join(model_ckpt, 'training_args.bin'))

    def _load_tokenizer(self, model_ckpt: str):
        return AutoTokenizer.from_pretrained(self.args.model_name_or_path)

    def _load_model(self, model_ckpt: str):
        # Check whether model exists
        if not os.path.exists(model_ckpt):
            raise Exception("Model doesn't exists! Train first!")
            
        try:
            # Config will be automatically loaded from model_dir
            model = AutoModelForSequenceClassification.from_pretrained(model_ckpt) 
    
            model.to("cuda")
            model.eval()
        except:
            raise Exception("Some model files might be missing...")
            
        return model

    @abstractmethod    
    def calculate(self, instances: List[str]) -> Tuple[Dict, float]:
        raise NotImplementedError

    def convert_input_file_to_tensor_dataset(self, lines,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):

        # Setting based on the current model type
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pad_token_id = self.tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        
        for line in lines:
            tokens = self.tokenizer.tokenize(line)
            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > self.args.max_seq_length - special_tokens_count:
                tokens = tokens[:(self.args.max_seq_length - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # for interpret functions
            ref_ids = [input_ids[0]] + [pad_token_id] * len(input_ids[1:-1]) + [input_ids[-1]]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.args.max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)

        # Change to Tensor
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

        #dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        dataset = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'token_type_ids': all_token_type_ids,
        }
        
        return dataset


class EmpIntentCalculator(ModelCalculator):
    LABELS = ['agreeing', 'acknowledging', 'encouraging', 'consoling', 'sympathizing', 'suggesting', 'questioning', 'wishing', 'neutral']
    
    def __init__(self, model_ckpt: str, model_name: str):
        super().__init__(model_ckpt, model_name)

        self.id2label = {i: empintent for i, empintent in enumerate(self.LABELS)}

    def calculate(self, instances: List[str]) -> Tuple[Dict, float]:
        metric_value = self._calculate(instances)
        return metric_value
    
    def _calculate(self, instances: List[str]) -> Tuple[Dict, float]:
        results = {'pred_intent': [], 'gold_intent': []}
        for instance in tqdm(instances, total=len(instances)):

            pred_resp = instance["model_response"].strip()
            gold_resp = instance["golden_response"].strip()

            input_data = [pred_resp] + [gold_resp]
            dataset = self.convert_input_file_to_tensor_dataset(input_data)

            batch = tuple(t.to('cuda') for k, t in dataset.items())

            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": None}

                outputs = self.model(**inputs)
                logits = outputs[0]

                logits = torch.nn.functional.softmax(logits, dim=1)

                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)

            assert len(preds) == 2, "output must be 2 length of"

            results['pred_intent'].append(self.id2label[int(preds[0])])
            results['gold_intent'].append(self.id2label[int(preds[1])])

        assert len(results['pred_intent']) == len(results['gold_intent'])
        acc = evaluate_acc(results['pred_intent'], results['gold_intent'])
        return results, acc

class EmotionCalculator(ModelCalculator):
    LABELS = ['proud', 'apprehensive', 'disappointed', 'faithful', 'impressed', 'devastated', 'prepared', 'nostalgic', 'annoyed', 'grateful', 'joyful', 'terrified', 'caring', 'trusting', 'sad', 'guilty', 'sentimental', 'hopeful', 'confident', 'surprised', 'furious', 'afraid', 'jealous', 'excited', 'lonely', 'disgusted', 'embarrassed', 'angry', 'content', 'ashamed', 'anticipating', 'anxious']
    
    def __init__(self, model_ckpt: str, model_name: str):
        super().__init__(model_ckpt, model_name)

        self.id2label = {i: emotion for i, emotion in enumerate(self.LABELS)}

    def calculate(self, instances: List[str]) -> Tuple[Dict, float]:
        metric_value = self._calculate(instances)
        return metric_value
    
    def _calculate(self, instances: List[str]) -> Tuple[Dict, float]:
        results = {'pred_emotion': [], 'gold_emotion': []}
        for instance in tqdm(instances, total=len(instances)):

            pred_resp = instance["model_response"].strip()
            gold_resp = instance["golden_response"].strip()

            input_data = [pred_resp] + [gold_resp]
            dataset = self.convert_input_file_to_tensor_dataset(input_data)

            batch = tuple(t.to('cuda') for k, t in dataset.items())

            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": None}

                outputs = self.model(**inputs)
                logits = outputs[0]

                logits = torch.nn.functional.softmax(logits, dim=1)

                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)

            assert len(preds) == 2, "output must be 2 length of"

            results['pred_emotion'].append(self.id2label[int(preds[0])])
            results['gold_emotion'].append(self.id2label[int(preds[1])])

        assert len(results['pred_emotion']) == len(results['gold_emotion'])
        acc = evaluate_acc(results['pred_emotion'], results['gold_emotion'])
        return results, acc

class EpitomeCalculator(ModelCalculator):
    LABELS = None

    def __init__(self, model_ckpt: str, model_name: str):
        from evaluation.modules.empathy_scorer import EmpathyScorer

        opt = {}
        opt['epitome_save_dir'] = model_ckpt
        self.model = EmpathyScorer(opt, batch_size=1, cuda_device="cuda")
        self.model_name = model_name

    def calculate(self, instances: List[str]) -> Dict:
        metric_value = self._calculate(instances)
        return metric_value

    def _calculate(self, instances: List[str]) -> Dict:
        pred_results = {
            'IP': [], 'EX': [], 'ER': [],
            'diff-IP': [], 'diff-EX': [], 'diff-ER': []
        }
        gold_results = {'IP': [], 'EX': [], 'ER': []}

        for instance in tqdm(instances, total=len(instances)):
            
            pred_resp = instance["model_response"].strip()
            gold_resp = instance["golden_response"].strip()

            pred_score = self.model([prev_input], [pred_resp])
            gold_score = self.model([prev_input], [gold_resp])

            for ep in ['IP', 'EX', 'ER']:
                pred_results[ep] += pred_score[ep][0]

                pred_results[f'diff-{ep}'].append(
                    math.pow(abs(pred_score[ep][0][0] - gold_score[ep][0][0]), 2)
                )

                gold_results[ep] += gold_score[ep][0]
        
        # average epitome score & diff-epitome score
        total_results = {}
        for k, v in pred_results.items():
            total_results[k] = sum(v)/len(v)
        
        for k, v in gold_results.items():
            total_results[f'gold-{k}'] = sum(v)/len(v)
        
        return total_results

