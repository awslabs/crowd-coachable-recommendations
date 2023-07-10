import torch
from torch.nn.parallel.data_parallel import (
    replicate as _replicate,
    DataParallel as _DataParallel,
)


class DataParallel(_DataParallel):
    def cache_replicas(self):
        print("caching replicas")
        self._replicas = _replicate(
            self.module, self.device_ids, detach=True
        )  # detach if no_grad
        return self

    def replicate(self, module, device_ids):
        if hasattr(self, "_replicas"):
            device_inds = [self.device_ids.index(d) for d in device_ids]
            return [self._replicas[r] for r in device_inds]
        return super().replicate(module, device_ids)
