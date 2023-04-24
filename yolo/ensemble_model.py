from ultralytics.nn.tasks import DetectionModel, BaseModel

class EnsembleModel(BaseModel):
    def __init__(self, cfgs, ch=3, nc=None, verbose=True):
        self.models = [DetectionModel(cfg, ch=ch, nc=nc, verbose=verbose) for cfg in cfgs]

    def predict():
        pass