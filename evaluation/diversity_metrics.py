from typing import List
from typing import Tuple

from nltk.tokenize.casual import casual_tokenize
from nltk.util import ngrams


class DistKCalculator:
    def __init__(self, k: int):
        self.k = k
    
    def calculate(self, instances: List[str]) -> Tuple[float, int]:
        metric_value = self._calculate(instances)
        return metric_value
    
    def _calculate(self, instances: List[str]) -> Tuple[float, int]:
        tokenized_responses = [
            casual_tokenize(instance) for instance in instances
        ]
        num_all_ngrams = 0
        all_ngram_set = set()

        for tokens in tokenized_responses:
            token_ngrams = list(ngrams(tokens, self.k))
            num_all_ngrams += len(token_ngrams)
            all_ngram_set.update(token_ngrams)

        dist_score = len(all_ngram_set) / num_all_ngrams
        return dist_score, len(all_ngram_set)


class Dist1Calculator(DistKCalculator):
    def __init__(self):
        super().__init__(k=1)


class Dist2Calculator(DistKCalculator):
    def __init__(self):
        super().__init__(k=2)


class Dist3Calculator(DistKCalculator):
    def __init__(self):
        super().__init__(k=3)