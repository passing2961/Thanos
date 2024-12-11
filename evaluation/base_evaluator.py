


class BaseEvaluator(object):
    def __init__(self, eval_result_dir):

        self.results = self._load_eval_results(eval_result_dir)

    def _load_eval_results(self, result_dir):
        with open(result_dir, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def predict(self, )
