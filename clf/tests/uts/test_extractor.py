from src.extractors import EXTRCT_REGISTRY
import pytest


@pytest.mark.parametrize(
    "extractor", ["EfficientNetExtractor", "SENetExtractor", "LangExtractor"]
)
def test_extractors(tmp_path, extractor):
    args = {
        "EfficientNetExtractor": {"version": 0, "from_pretrained": True},
        "SENetExtractor": {"version": "senet154", "from_pretrained": None},
        "LangExtractor": {"pretrained": "bert-base-uncased"},
    }
    assert extractor in args, f"{extractor} not in testing model args, please add it"
    model = EXTRCT_REGISTRY.get(extractor)
    model = model(**args[extractor])
