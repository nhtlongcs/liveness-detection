from allennlp.predictors.predictor import Predictor


class SRLPredictor(object):
    def __init__(self, path):
        self.predictor = Predictor.from_path(str(path))

    def run_sentence(self, query: str):
        return self.predictor.predict(sentence=query)
