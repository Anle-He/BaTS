from .AbstractRunner import AbstractRunner


class LTSFRunner(AbstractRunner):
    def __init__(
            self,
            cfg: dict,
            device,
            scaler,
            log=None):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.scaler = scaler
        self.log = log

        self.clip_grad = cfg['OPTIM'].get('clip_grad')