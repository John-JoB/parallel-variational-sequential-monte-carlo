class OptimList:
    def __init__(self, optims):
        super().__init__()
        self.optims = optims

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def step(self):
        for optim in self.optims:
            optim.step()

class LRSchedualrList:
    def __init__(self, lrs):
        super().__init__()
        self.lrs = lrs

    def step(self):
        for lr in self.lrs:
            lr.step()