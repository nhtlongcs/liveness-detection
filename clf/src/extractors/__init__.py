from registry import Registry

EXTRCT_REGISTRY = Registry("EXTRACTOR")

from .eff_net import *
from .senet import *
from .hugging import *
from .clip.clip_extractors import *