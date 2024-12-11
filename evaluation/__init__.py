from .diversity_metrics import (
    Dist1Calculator,
    Dist2Calculator,
    Dist3Calculator
)
from .empathy_metrics import(
    EmpIntentCalculator,
    EmotionCalculator,
    #EpitomeCalculator
)

_metrics = {
    'dist-1': Dist1Calculator,
    'dist-2': Dist2Calculator,
    'dist-3': Dist3Calculator,
    'empintent': EmpIntentCalculator,
    'emotion': EmotionCalculator,
    #'epitome': EpitomeCalculator
}