import os
import nltk

# nltk.download("punkt")
# nltk.download("wordnet")

from pytorch_lightning import seed_everything

TEST_ROOT = os.path.realpath(os.path.dirname(__file__))
PACKAGE_ROOT = os.path.dirname(TEST_ROOT)
DATASETS_PATH = os.path.join(PACKAGE_ROOT, "data")
# generate a list of random seeds for each test
ROOT_SEED = 1234


def reset_seed():
    seed_everything()
