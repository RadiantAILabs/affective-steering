"""Qwen3.5-4B (8-bit MLX) as the target: batched chat generation with optional steering (at one layer or several),
per-layer residual readouts over a transcript, and the pre-reply state used to gate steering per turn."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm import batch_generate, load
from mlx_lm.models.base import create_attention_mask

from mlx_helpers import DEFAULT_MLX_MODEL, limit_memory, text_layers

READ_LAYERS = list(range(8, 30, 2))  # residual stream after these decoder layers


class Steer(nn.Module):
    """Adds a fixed vector to one layer's MLP output, i.e. to the residual stream leaving that layer."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.vec = None

    def __call__(self, x):
        out = self.inner(x)
        return out if self.vec is None else out + self.vec.astype(out.dtype)


class Target:
    def __init__(self, mem_gb=9):
        limit_memory(mem_gb)
        self.model, self.tok = load(DEFAULT_MLX_MODEL)
        self.layers = text_layers(self.model)
        self.steers = {}

    def set_steering(self, layer=None, vec=None):
        """set_steering(layer, vec) adds vec to the residual stream leaving `layer`; set_steering({layer: vec, ...})
        does the same at several layers at once (multi-layer steering); set_steering() turns all steering off."""
        for st in self.steers.values():
            st.vec = None
        if layer is None:
            return
        pushes = layer if isinstance(layer, dict) else {layer: vec}
        for l, v in pushes.items():
            l = int(l)
            if l not in self.steers:
                self.steers[l] = Steer(self.layers[l].mlp)
                self.layers[l].mlp = self.steers[l]
            self.steers[l].vec = mx.array(np.asarray(v, np.float32))

    def pre_state(self, conversations, layer):
        """Residual stream after `layer` at the last prompt token, as the model sees it right before it generates the
        next reply (generation prompt included, steering off). Returns [len(conversations), H]."""
        saved = {l: st.vec for l, st in self.steers.items()}
        for st in self.steers.values():
            st.vec = None
        out = []
        try:
            for m in conversations:
                h = self.model.language_model.model.embed_tokens(mx.array([self.ids(m)]))
                mask = create_attention_mask(h, None)
                for lyr in self.layers[: layer + 1]:
                    h = lyr(h, mask=None if lyr.is_linear else mask, cache=None)
                out.append(np.array(h[0, -1].astype(mx.float32)))
                mx.clear_cache()
        finally:
            for l, v in saved.items():
                self.steers[l].vec = v
        return np.stack(out)

    def ids(self, messages, add_generation_prompt=True):
        r = self.tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                         enable_thinking=False, tokenize=True)
        return list(r["input_ids"] if hasattr(r, "keys") else r)

    def reply_spans(self, full):
        """(content_start, content_end) of each assistant message, found from the special tokens. The template drops
        the empty <think></think> block from earlier turns and keeps it on the last, so prefix lengths don't line up.
        content_start - 1 is the last token before the reply: the model's state as it starts to answer."""
        t = self.tok
        im_start, im_end = t.convert_tokens_to_ids("<|im_start|>"), t.convert_tokens_to_ids("<|im_end|>")
        think_end = t.convert_tokens_to_ids("</think>")
        asst = t.encode("assistant", add_special_tokens=False)
        spans = []
        for i, x in enumerate(full):
            if x == im_start and full[i + 1 : i + 1 + len(asst)] == asst:
                j = i + 1 + len(asst) + 1  # past "assistant\n"
                if think_end in full[j : j + 4]:
                    j = full.index(think_end, j) + 2  # past "</think>\n\n"
                e = full.index(im_end, j)
                spans.append((j, e))
        return spans

    def generate(self, conversations, max_tokens=220):
        """conversations: list of message lists. Greedy; one reply each."""
        prompts = [self.ids(m) for m in conversations]
        out = batch_generate(self.model, self.tok, prompts, max_tokens=max_tokens, prefill_batch_size=2, prefill_step_size=512)  # bounded prefill memory on long transcripts
        mx.clear_cache()
        return [t.strip() for t in out.texts]

    def readout(self, messages):
        """Residual stream (READ_LAYERS) for a finished transcript. Per assistant turn returns
        pre: the activation at the last token before the reply starts (the model's state as it decides how to answer)
        resp: the mean over the reply's tokens. Shapes [turns, len(READ_LAYERS), H]."""
        # a trailing placeholder turn makes every reply "historical", so all render alike (no <think> block)
        full = self.ids(list(messages) + [{"role": "user", "content": "."}], add_generation_prompt=False)
        spans = self.reply_spans(full)
        assert len(spans) == sum(m["role"] == "assistant" for m in messages), "reply spans don't match the transcript"
        h = self.model.language_model.model.embed_tokens(mx.array([full]))
        mask = create_attention_mask(h, None)
        pre = np.zeros((len(spans), len(READ_LAYERS), h.shape[-1]), np.float32)
        resp = np.zeros_like(pre)
        for li, layer in enumerate(self.layers[: max(READ_LAYERS) + 1]):
            h = layer(h, mask=None if layer.is_linear else mask, cache=None)
            mx.eval(h)
            if li in READ_LAYERS:
                k = READ_LAYERS.index(li)
                hn = np.array(h[0].astype(mx.float32))
                for t, (s, e) in enumerate(spans):
                    pre[t, k] = hn[s - 1]
                    resp[t, k] = hn[s:e].mean(0)
        mx.clear_cache()
        return pre, resp
