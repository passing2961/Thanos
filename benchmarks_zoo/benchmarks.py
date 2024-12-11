#from .wrappers import *
import os
import json

import torchvision

from benchmarks_zoo.wrappers import *
from benchmarks_zoo import register_benchmark
from utils.common import *


@register_benchmark("bst")
def bst(config):
    return BSTDataset(config)

@register_benchmark("empathy")
def empathy(config):
    return EmpathyDataset(config)

@register_benchmark("dailydialog")
def dailydialog(config):
    return DailyDialogDataset(config)

@register_benchmark("prosocial")
def prosocial(config):
    return ProsocialDataset(config)

@register_benchmark("ours")
def ours(config):
    return OursDataset(config)

@register_benchmark("photochat")
def photochat(config):
    return PhotoChatDataset(config)