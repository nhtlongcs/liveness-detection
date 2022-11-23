class HybridLrScheduler():
    """ Wrapper Class around lr_scheduler to return a 'dummy' optimizer to pass
        pytorch lightning checks
    """

    def __init__(self, hybrid_optimizer, idx, lr_scheduler) -> None:
        self.optimizer = hybrid_optimizer
        self.idx = idx
        self.lr_scheduler = lr_scheduler

    def __getattribute__(self, __name: str):
        if __name in {"optimizer", "idx", "lr_scheduler"}:
            return super().__getattribute__(__name)
        else:
            return self.lr_scheduler.__getattribute__(__name)
