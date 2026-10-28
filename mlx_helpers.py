"""The few MLX helpers the target needs, vendored so this repository stands alone.

Taken from sweep/mlx_qwen.py in https://github.com/hsehskad/neuron-explorer (branch mlx-apple-silicon), our Apple
Silicon port of https://github.com/lyramakesmusic/neuron-explorer. memguard.py comes from the same place.
"""
import os

import mlx.core as mx

# made with: mlx_lm.convert --hf-path Qwen/Qwen3.5-4B -q --q-bits 8 --mlx-path ~/.cache/mlx-models/Qwen3.5-4B-q8
# (set MLX_MODEL to use a different path or an HF repo id)
DEFAULT_MLX_MODEL = os.environ.get("MLX_MODEL", os.path.expanduser("~/.cache/mlx-models/Qwen3.5-4B-q8"))


def limit_memory(gb):
    """Caps MLX. Past the memory limit MLX waits for queued work to free buffers before allocating more, and the cache
    limit stops it hoarding freed buffers. Neither is a hard ceiling (MLX still swaps rather than fail), so long runs
    also go under memguard.py. The default limit is 1.5x the GPU working set, 17 GB on a 16 GB M1."""
    mx.set_memory_limit(int(gb * 2**30))
    mx.set_cache_limit(int(min(1.0, gb / 8) * 2**30))


def text_layers(model):
    return model.language_model.model.layers
