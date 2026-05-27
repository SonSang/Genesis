"""Running mean / std normalization.

Used by SHAC to whiten observations before feeding to the actor/critic. Tracks
mean and variance via Welford's online algorithm; `normalize(x)` returns
`(x - mean) / sqrt(var + eps)` and clamps via the fixed range used by DiffRL.
"""

import torch


class RunningMeanStd:
    def __init__(self, shape, device, epsilon: float = 1e-4):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = torch.tensor(epsilon, device=device, dtype=torch.float32)
        self._frozen = False

    def update(self, x: torch.Tensor):
        if self._frozen:
            return
        with torch.no_grad():
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]

            delta = batch_mean - self.mean
            tot_count = self.count + batch_count
            self.mean = self.mean + delta * (batch_count / tot_count)
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
            self.var = M2 / tot_count
            self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-5)

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]
