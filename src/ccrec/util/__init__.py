import collections, contextlib
import numpy as np


def merge_unique(list_of_lists, num_per_list, total, rng=np.random):
    for a in list_of_lists:
        assert len(a) >= total, f"please provide enough inputs to avoid short returns {a}"

    random_groups = rng.permutation([i for i, a in enumerate(list_of_lists) for _ in a])
    list_of_queues = [collections.deque(a) for a in list_of_lists]

    c = collections.Counter()
    unique = collections.OrderedDict()
    nunique = 0
    for i in random_groups:
        x = list_of_queues[i].popleft()
        if c[i] < num_per_list[i] and nunique < total and unique.get(x, i) == i:
            assert x not in unique, f"duplication detected in list {i}: {list_of_lists[i]}"
            c[i] += 1
            nunique += 1
            unique[x] = i

    return list(unique.keys()), list(unique.values())


@contextlib.contextmanager
def _device_mode_context(module, device, training):
    old_device = getattr(module, "device", 'cpu')
    old_training = module.training
    module.to(device)
    module.train() if training else module.eval()
    yield module
    module.to(old_device)
    module.train() if old_training else module.eval()
