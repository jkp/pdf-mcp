import os

# Constrain PyTorch CPU threads at the process level, before any model loads.
# Without this, torch defaults to ALL cores (e.g. 16 on a 24-core box),
# causing 400%+ CPU and thermal throttling during inference.
_torch_threads = os.environ.get("TORCH_THREADS") or str(min(4, os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", _torch_threads)
os.environ.setdefault("MKL_NUM_THREADS", _torch_threads)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import torch

    torch.set_num_threads(int(_torch_threads))
    torch.set_num_interop_threads(int(_torch_threads))
except ImportError:
    pass
