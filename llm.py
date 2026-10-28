"""Claude on Amazon Bedrock: the tester (user simulator), the judges and the persona-artifact generator.

Credentials come from the usual AWS chain (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN, AWS_PROFILE
or an instance role); nothing is stored here. AWS_REGION picks the Bedrock region, and the model ids can be overridden
with OPUS_MODEL, SONNET_MODEL, HAIKU_MODEL and OPUS46_MODEL.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
from anthropic import AnthropicBedrock

OPUS = os.environ.get("OPUS_MODEL", "global.anthropic.claude-opus-5-5")
SONNET = os.environ.get("SONNET_MODEL", "global.anthropic.claude-sonnet-5-5")
HAIKU = os.environ.get("HAIKU_MODEL", "global.anthropic.claude-haiku-5-5")
OPUS46 = os.environ.get("OPUS46_MODEL", "global.anthropic.claude-opus-4-6-v1")

_client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "eu-west-2"), max_retries=6, timeout=600)
_usage_lock = threading.Lock()
USAGE = {}  # model -> [input_tokens, output_tokens, calls]


def chat(model, messages, system=None, max_tokens=16000, retries=4):
    """One Messages call; returns the final text. Retries transient errors beyond the SDK's own retries."""
    kw = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kw["system"] = system
    for attempt in range(retries):
        try:
            r = _client.messages.create(**kw)
            break
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    with _usage_lock:
        u = USAGE.setdefault(model, [0, 0, 0])
        u[0] += r.usage.input_tokens
        u[1] += r.usage.output_tokens
        u[2] += 1
    if r.stop_reason == "refusal":
        raise RuntimeError(f"{model} refused")
    return "".join(b.text for b in r.content if b.type == "text")


def parse_json(text):
    """The last fenced or bare JSON object/array in a reply."""
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
    for cand in reversed(fenced + [text]):
        cand = cand.strip()
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = cand.find(opener), cand.rfind(closer)
            if i != -1 and j > i:
                try:
                    return json.loads(cand[i : j + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no JSON in reply: {text[:300]!r}")


def chat_json(model, messages, system=None, max_tokens=16000, tries=3):
    last = None
    for _ in range(tries):
        text = chat(model, messages, system, max_tokens)
        try:
            return parse_json(text)
        except ValueError as e:
            last = e
    raise last


def pmap(fn, items, workers=12):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))
