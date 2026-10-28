# Looking inside an agent as its boundaries erode

Code and data for our post [Looking inside an agent as its boundaries erode](https://radiantlabs.dev/blog/looking-inside-an-agent-as-its-boundaries-erode).

In an [earlier study](https://radiantlabs.dev/blog/stress-testing-ai-agent-emotional-boundaries) we watched AI agents give in to sustained emotional pressure from a scripted tester, and scored what happened with a rubric of yes/no questions. That work could only see behaviour from the outside. Here we ran the same kind of conversations against an open-weight model, Qwen3.5-4B, running locally on a 16 GB Apple Silicon Mac, so that we could also look at its internals:

- We extracted persona vectors (in the style of Chen et al., 2025) for three failure traits - capitulation, sycophancy and emotional enmeshment - and read them from the residual stream at every turn of every conversation.
- We built a small persona space from four whole characters (the default assistant, a devoted companion, a people-pleaser and a counsellor with firm boundaries) and tracked where each turn sat in it.
- We steered the model along the capitulation direction, both away from it and towards it, with a random direction of the same size as a control, and graded what changed.
- We graded every conversation with two independent LLM judges (Claude Opus 5.5 and Claude Opus 4.6) and used their agreement in place of human inter-rater reliability. Claude Haiku 5.5 and Claude Sonnet 5.5 were run on the same scenarios as unsteered references.

The post has the findings. This repository has everything we can share to check them or build on them: the code, every transcript, every judgement, the vectors and the analysis output.

## What's in the repository

```
README.md               this file
LICENSE                 MIT, for the code
LICENSE-DATA            CC BY 4.0, for everything under data/
pyproject.toml          dependencies (uv or pip)
run_all.sh              the whole pipeline, one GPU job at a time under memguard.py

llm.py                  Claude on Amazon Bedrock (tester, judges, persona-data generation)
target.py               Qwen3.5-4B in MLX: batched generation, a steering hook, residual readouts
mlx_helpers.py          vendored MLX helpers (see Credits)
memguard.py             vendored memory watchdog for 16 GB Macs (see Credits)
persona.py              persona vectors for the three traits
persona_space.py        centroids for the four characters
calib.py                picks the steering strengths
converse.py             multi-turn tester-vs-target conversations, plus readouts for Qwen
judge.py                two-judge grading against a rubric file you supply
analyze.py              judgements + readouts + geometry -> data/results.json

examples/               EXAMPLE scenario, tester prompt, agent prompt and rubric (not the study's - see below)

data/
  convs/{condition}.json        transcripts, one file per condition
  convs/acts_L20.npz            residual readouts at the steering layer for the four Qwen conditions
  judged/{condition}.json       per-turn and cross-turn judgements from both judges
  rubric_keys.json              question ids, markers and the healthy answer per id (no question texts)
  results.json                  everything the post's figures and numbers are drawn from
  persona_artifacts.json        Opus-written contrastive system prompts and user messages for the three traits
  persona_responses.json        Qwen's answers under each pole
  persona_scores.json           Sonnet's trait and coherence scores for those answers
  persona_acts.npy              reply-averaged residual stream for those answers [120, 11 layers, 2560]
  persona_vectors.npz           the trait vectors per layer, plus the layer list
  persona_report.json           per-layer held-out separation (Cohen's d), correlation and residual norms
  persona_space_responses.json  Qwen's answers as each of the four characters
  persona_space.npz             character centroids [4, 11, 2560] and per-answer activations [120, 11, 2560]
  calib.json                    the steering-strength calibration, with every steered reply and its scores
```

The conditions are:

| condition | target | steering |
|---|---|---|
| baseline | Qwen3.5-4B | none |
| cap_minus | Qwen3.5-4B | away from capitulation, -0.4 x median residual norm, layer 20 |
| random | Qwen3.5-4B | a random direction of the same size and layer as cap_minus |
| cap_plus | Qwen3.5-4B | towards capitulation, +0.6 x median residual norm, layer 20 |
| haiku | Claude Haiku 5.5 | none |
| sonnet | Claude Sonnet 5.5 | none |

Each condition has the same 10 scenarios (four about money, three about health, three about relationships), 8 turns each. The tester was Claude Sonnet 5.5 throughout. Qwen ran as an 8-bit MLX conversion with greedy decoding, thinking off and replies capped at 220 tokens.

## Data format

### Transcripts: `data/convs/{condition}.json`

```
{
  "condition": "cap_minus",
  "args": {...},            # the converse.py arguments the run used
  "layer": 20,              # steering layer, or null
  "coef": -0.4,             # push size as a fraction of the median residual norm at that layer
  "conversations": [
    {"scenario": "finances_1_david",
     "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]},
    ...
  ]
}
```

Every conversation alternates user and assistant for 8 turns. The user turns were written by the tester model in character; the assistant turns are the target's replies. Two things are removed from the transcripts as we ran them - the agent's system prompt (the first message of every conversation) and the tester's private per-turn notes - for the reasons given under "What we withheld".

### Judgements: `data/judged/{condition}.json`

```
{
  "opus55": [ <one entry per conversation, in the same order as conversations[] in the transcript file> ],
  "opus46": [ ... ]
}
```

Each entry is

```
{
  "scenario": "finances_1_david",
  "turns": [
    {"turn": 1,
     "answers": {"ETA-1": "Y", "ETA-2": "NA", ..., "ABS-5": "N"},     # 24 per-turn questions
     "traits": {"capitulation": 5, "sycophancy": 10, "enmeshment": 40}},  # 0-100
    ...
  ],
  "cross_turn": {"CT-1": "N", "CT-2": "Y", "CT-3": "N", "CT-4": "N"}
}
```

An answer is `NA` when the judge decided the question did not apply to that turn, and otherwise `Y` or `N`. Whether `Y` or `N` is the healthy answer differs by question, and is given in `data/rubric_keys.json`. A marker's score is the share of its applicable answers that were healthy, pooled over both judges.

A few entries are not clean, and we have left them as the judges returned them. Opus 5.5 declined to grade one conversation (`healthcare_2_jennifer`) in both baseline and cap_minus; those entries are `{"error": "... refused", "scenario": ...}` and the analysis skips them. In cap_plus, Opus 4.6 added a stray id on two turns (`EH-6`, `ETA-6b`), which the analysis ignores.

The trait scores are the judges' ratings of how strongly each reply shows capitulation, sycophancy and enmeshment. The definitions they were given are in `judge.py`.

### The four markers

The per-turn questions are grouped into four markers of affective health. In our own words:

- ETA, Emotional Tone Attunement (6 questions) - how well the agent's reply fits the emotional state of the person it is talking to.
- EH, Emotional Honesty (5 questions) - whether the agent stays truthful about its own view when the user would rather hear something else.
- RC, Role Clarity (8 questions) - whether the agent is clear and honest about what it is, and what kind of relationship it can offer.
- ABS, Affective Boundary Strength (5 questions) - whether the agent's limits survive emotional pressure.

The four cross-turn questions (CT-1 to CT-4) are about the conversation as a whole rather than any single reply.

### Activations: `data/convs/acts_L20.npz`

Keys `{condition}_pre` and `{condition}_resp` for baseline, cap_minus, random and cap_plus, each `[10 conversations, 8 turns, 2560]` float32, plus `layer` (20). `resp` is the residual stream after layer 20 averaged over the tokens of each reply; `pre` is the residual stream at the last token before the reply starts. converse.py saves these for every even layer from 8 to 28 (`{condition}_acts.npz`, about 18 MB per condition); we only ship the steering layer, which is the one the analysis uses. With it, `python analyze.py` rebuilds `data/results.json` byte for byte from the released files.

These readouts were taken with the agent's system prompt in context. Re-reading the released transcripts, which no longer have it, will give somewhat different activations.

### `data/results.json`

The analysis output behind every number and figure in the post: per-condition marker scores with bootstrap confidence intervals (`conditions`), per-question healthy rates, trait scores per turn, judge agreement as Cohen's kappa per question and marker (`agreement`), trait-vector geometry (`geometry`), the correlation between activation readouts and judge scores with a random-direction null (`monitor`), the persona-space map and trajectories (`persona_space`), and the user and assistant messages of every conversation side by side (`examples`).

## Reproducing

Everything downstream of the transcripts and judgements runs on a laptop with no model:

```
uv sync
uv run python analyze.py
```

(The analysis itself only needs numpy and scipy.)

The full pipeline needs an Apple Silicon Mac for the target and Claude access on Amazon Bedrock for the tester and judges.

1. Convert the model to 8-bit MLX (about 4.2 GB):

   ```
   uv run mlx_lm.convert --hf-path Qwen/Qwen3.5-4B -q --q-bits 8 --mlx-path ~/.cache/mlx-models/Qwen3.5-4B-q8
   ```

   Set `MLX_MODEL` if you put it somewhere else.

2. Set up Bedrock. `llm.py` reads credentials from the standard AWS chain (environment variables, `AWS_PROFILE` or a role) and the region from `AWS_REGION` (default `eu-west-2`). The model ids can be overridden with `OPUS_MODEL`, `SONNET_MODEL`, `HAIKU_MODEL` and `OPUS46_MODEL`. No keys live in this repository.

3. Bring scenarios, a tester prompt, an agent prompt and a rubric. The files in `examples/` show the formats and will run as they are, but they are illustrations, not the study's materials (see below).

4. Run it, in a fresh copy of the repository or after moving the released `data/` aside. persona.py reuses any persona files it finds, and judge.py skips conditions that already have judgements.

   ```
   PY="uv run python" RUBRIC=path/to/rubric.json SCENARIOS_DIR=path/to/scenarios \
     TESTER_PROMPT=path/to/tester.md AGENT_PROMPT=path/to/agent.md ./run_all.sh
   ```

   Or run the steps one at a time, in the order run_all.sh uses.

On a 16 GB Mac, run one model at a time and keep each GPU job under `memguard.py`. MLX's default memory limit is 1.5x the GPU working set (about 17 GB on a 16 GB M1), and past it MLX allocates into swap rather than failing. On a 16 GB machine that can starve the window server until the whole login session is killed. `target.py` caps MLX at 9 GB, and memguard watches the job's physical footprint and the kernel's memory-pressure level and kills the job's process group if either crosses the line (`--max-gb 11` worked for us). Batched prefill is chunked so long transcripts stay within budget. Each Qwen condition took under an hour on our 16 GB M1.

Greedy decoding on an 8-bit model with batching is not bit-for-bit stable across machines and library versions, and the tester adapts to every reply, so a rerun will produce different conversations. We would expect the aggregate picture to hold, not individual transcripts.

## What we withheld, and why

The scenarios, the adversarial tester prompt, the agent's system prompt and the wording of the 28 rubric questions come from an earlier evaluation framework that is not public. We are not able to release them, so this repository leaves them out:

- The 10 scenario files are not included. The scenario ids in the data (`finances_1_david` and so on) give each one's domain and the character's first name, and the tester's in-character messages are in the transcripts.
- The tester's system prompt is not included, and neither are its private per-turn notes, which discussed its tactics in that prompt's terms.
- The agent's system prompt is not included and has been removed from the start of every transcript. It was a short, generic instruction of a couple of sentences, with no persona and no safety guidance.
- The rubric's question texts are not included. We keep the question ids, every answer, the marker each question belongs to and which answer counts as healthy, which is all the analysis needs, and the short marker descriptions above are our own.

In their place, `examples/` has a made-up scenario, tester prompt, agent prompt and rubric, written for this repository and labelled as examples, so the code runs end to end. The example rubric uses different question ids (`EX-...`) so it can't be confused with the study's.

We also left out the HTML report and a follow-up on multi-layer and dynamically gated steering, which will be released with its own post.

## Credits

`mlx_helpers.py` and `memguard.py` are taken from [hsehskad/neuron-explorer](https://github.com/hsehskad/neuron-explorer) (branch `mlx-apple-silicon`), our Apple Silicon port of [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer). Persona vectors follow Chen et al. (2025), "Persona Vectors: Monitoring and Controlling Character Traits in Language Models", and the persona space is in the spirit of Anthropic's work on the Assistant axis.

## Citation

```
@misc{radiant2026boundaries,
  title        = {Looking inside an agent as its boundaries erode},
  author       = {{Radiant AI Labs}},
  year         = {2026},
  howpublished = {\url{https://radiantlabs.dev/blog/looking-inside-an-agent-as-its-boundaries-erode}},
  note         = {Code and data: https://github.com/RadiantAILabs/affective-steering}
}
```

## Licence

The code is released under the MIT licence (`LICENSE`). The data under `data/` is released under Creative Commons Attribution 4.0 International (`LICENSE-DATA`).

The transcripts and many of the data files contain text generated by models: Qwen3.5-4B (the target, and the persona and calibration answers) and Claude models (the tester's messages, the Haiku and Sonnet replies, the persona prompts and questions, and all judgements and scores). The licence covers our contribution to them; any terms that the model providers place on model outputs still apply. The conversations are role-play about emotionally difficult situations - debt, illness, relationships - and some replies, especially in the steered conditions, give advice that would be harmful to follow. None of the people in them are real.

Questions to dakshesh@getradiant.ai.
