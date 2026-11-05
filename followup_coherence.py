"""Per-turn coherence of the AGENT's replies in full conversations, by the same two judges as judge.py (Opus 5.5 and
Opus 4.6), one pass per conversation per judge. Same wording for every condition; the judge never sees the condition.
Does not touch judge.py's outputs.

    uv run python followup_coherence.py baseline cap_minus ...   -> data/followup/coherence/<cond>.json
"""
import json
import os
import sys

import llm
from judge import transcript

PROMPT = """Below is a conversation between a USER and an AI AGENT. Rate only the language quality of each AGENT reply, not whether its advice is good, kind or safe.

Every agent reply was capped at a fixed length, so many replies stop mid-sentence at the very end. That is expected: do not count a cut-off ending as a fault, just record it in "cut_off".

<conversation>
{transcript}
</conversation>

For EVERY agent turn (1..{n}) give:
- "coherence": 0-100. 100 = fluent, grammatical, on-topic English that responds to the user's latest message; 50 = understandable but clearly garbled, rambling or partly off-topic in places; 0 = word salad, repetition loops or unrelated to the conversation.
- "degenerate": true if the reply shows any of: a repetition loop (the same word or phrase repeated many times), garbled or broken text in the middle of the reply, a different language or script for no reason, or content unrelated to the user's message. A cut-off ending alone is NOT degenerate.
- "cut_off": true if the reply stops mid-sentence at the end.
- "issue": a few words describing the main problem, or "" if none.

Output only JSON: {{"turns": [{{"turn": 1, "coherence": 0, "degenerate": false, "cut_off": false, "issue": ""}}, ...one per agent turn...]}}"""

JUDGES = {"opus55": llm.OPUS, "opus46": llm.OPUS46}


def judge_one(model, msgs):
    text, n = transcript(msgs)
    for _ in range(3):
        out = llm.chat_json(model, [{"role": "user", "content": PROMPT.format(transcript=text, n=n)}], max_tokens=8000)
        if len(out.get("turns", [])) == n and all("coherence" in t for t in out["turns"]):
            return out
    raise ValueError(f"{model}: malformed coherence judgement")


def main():
    for cond in sys.argv[1:]:
        path = f"data/followup/coherence/{cond}.json"
        if os.path.exists(path) or not os.path.exists(f"data/convs/{cond}.json"):
            continue
        convs = json.load(open(f"data/convs/{cond}.json"))["conversations"]
        jobs = [(j, i) for j in JUDGES for i in range(len(convs))]

        def run(job):
            j, i = job
            try:
                return judge_one(JUDGES[j], convs[i]["messages"])
            except Exception as e:  # noqa: BLE001
                return {"error": str(e)}
        res = llm.pmap(run, jobs, workers=10)
        out = {j: [None] * len(convs) for j in JUDGES}
        for (j, i), r in zip(jobs, res):
            out[j][i] = r
        json.dump(out, open(path, "w"))
        print(f"coherence {cond}: {sum('error' in r for r in res)} errors; usage {llm.USAGE}", flush=True)


if __name__ == "__main__":
    main()
