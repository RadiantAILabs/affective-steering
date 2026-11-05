# Follow-up: multi-layer, cleaned and gated steering (re:dvx ideas) on Qwen3.5-4B

These are our working notes for the follow-up post, [Single-reply calibration underestimates the multi-turn cost of activation steering](https://radiantlabs.dev/blog/how-hard-to-steer-an-agents-boundaries). The post is the canonical write-up; this file keeps the full tables and the per-step detail behind it. File names refer to this repository.

Date: 28 Oct 2026. Model: Qwen3.5-4B 8-bit MLX on a 16 GB M1. The setup matches the original study: the study's 10 scenarios x 8 turns, a Sonnet 5.5 tester, and two judges (Opus 5.5, Opus 4.6). n = 10 conversations per condition throughout.

## Question

In the original study, a single-layer push away from capitulation (L20, coef -0.4 x median residual norm, push norm 4.5) left judged capitulation unchanged (57.4 baseline vs 54.8). Would three ideas from re:dvx do better? They are multi-layer steering, removing the component the trait vectors share, and per-turn gating on the pre-reply state. We could not consult a re:dvx implementation (the repository we were pointed to returned "not found" when we tried it), so each method below is our own implementation of the idea as we understood it.

## Headline

1. **Dose, not layers, is what moved capitulation.** A push away from raw capitulation spread over layers 12-28 (ml_away_raw) cut judged capitulation from 57.4 to 45.8. A single-layer L20 push of the same total norm (sl_matched) cut it further, to 37.6. The multi-layer version was gentler on the text (coherence 74.5 vs 67.8), but it was not more effective. The old single-layer null result came from the dose. The old calibration (single-turn prompts, where capitulation was at the floor, so it could only see the cost) capped the push at -0.4 (norm 4.5) because -0.6 lost 11 coherence points. At the larger dose capitulation moves, and coherence pays for it, as that calibration warned.
2. **Both pushes that worked also broke the text in long conversations.** Scored by the same two judges on the full conversations, coherence fell from 86.7 (baseline) to 74.5 (ml_away_raw) and 67.8 (sl_matched). Degenerate turns, mostly verbatim repeats of an earlier reply, went from 0% to 12.5% and 22.5%. Our calibration scored single replies and missed this.
3. **Separating "held the line" from "lost the thread".** On coherent turns only, compared with baseline at the same scenario and turn:
   - sl_matched still capitulates less (-11.8 [-19.7, -4.9], 9 of 10 scenarios, sign p = 0.02). This depends on the coherence cut-off: -14.7 (9/1) at 60, -11.8 (9/1) at 70, and -1.4 (4/2, only 28 turns left) at 80 (`data/followup/coherence_threshold_sensitivity.json`). It is not robust evidence that the model holds the line on fully fluent replies.
   - ml_away_raw's effect mostly disappears (-3.9 [-14.2, +9.4], 8/2, p = 0.11; -6.3 at a cut-off of 60, -0.9 at 80).
   - On incoherent turns both conditions score about 35 points lower than baseline. Part of the "improvement" is the model recycling an earlier refusal or losing the thread.
   - The same coherent-turn, turn-matched comparison for the other measures: ml_away_raw still lowers sycophancy (-11.2 [-22.7, -2.4], 9/1, p 0.02) and improves Emotional Honesty (8/1, p 0.04) and Tone Attunement (9/0, p 0.004). sl_matched gives sycophancy -16.2 [-26.7, -6.6] (8/2), EH 8/1 and ETA 8/1. ml_random_raw shows nothing (5/5, 6/4, 4/6). So on coherent turns the multi-layer push mainly changes how sycophantic and emotionally honest the replies are, more than whether the model capitulates (`data/followup/coherent_turn_other_metrics.json`).
4. **Removing the shared component backfired** (ml_away). Capitulation was unchanged at 59.9, and the push moved the model towards sycophancy and enmeshment (readout), with enmeshment judged 34.5 vs 27.1.
5. **Gating protects coherence but, as built, does too little.** dyn_gate_raw (the sycophancy gate driving the raw multi-layer push) kept coherence at 87.8 but left capitulation unchanged (63.0). The gate fired on 30 of 80 turns, almost all at half dose, so the average push was about a fifth of ml_away_raw's. The pre-reply gate signal itself is good (Spearman 0.72 with the next reply's judged sycophancy).
6. Random multi-layer pushes did nothing: ml_random (norm 6.7) gave capitulation 60.0, coherence 88.3. ml_random_raw (norm 10.0, the ml_away_raw size) kept coherence at 87.9 with 1 degenerate turn, so the text damage under ml_away_raw and sl_matched is specific to the capitulation direction, not to push size. Its capitulation came out higher than baseline (67.1, +9.8, 3/7 scenarios, p 0.34). That is probably noise, but it makes comparisons against this control look better than they are. Against it, ml_away_raw improves in 10 of 10 scenarios (p 0.002), and the comparison with baseline is the more conservative one.

## What was built

| File | Change |
|---|---|
| `followup_vectors.py` (new) | Unit trait vectors at the even layers 12-28. Per layer, projects out the mean direction of the three unit vectors and re-normalises. Saves `cap_clean`, `capsyc_clean`, the raw units (`capitulation_raw` etc.) and `random` (an independent unit vector per layer, seed 0) -> `data/followup/ml_vectors.npz`, `geometry.json` |
| `target.py` | `set_steering({layer: vec, ...})` for multi-layer pushes (the old `set_steering(layer, vec)` and `set_steering()` behave as before). New `pre_state(convs, layer)` is the residual at the last prompt token, with the generation prompt and steering off |
| `converse.py` | New flags `--ml <array>` (multi-layer push; per-layer vec = coef x median resid norm(l) x unit(l) / 9, so coef is a total budget over the 9 layers) and `--gate <json>` (per-turn gating; conversations bucketed into 0 / half / full, each bucket batched with one broadcast vector; gate log saved per conversation). Old invocations are unchanged |
| `followup_calib.py`, `followup_calib_raw.py`, `followup_calib_sl.py` (new) | Gate quantiles and dose sweeps on pressured multi-turn states -> `data/followup/calib*.json`, `gate.json` |
| `followup_coherence.py` (new) | Per-turn coherence (0-100), a degenerate flag and a cut-off flag on full conversations, same two judges, condition-blind, identical prompt for every condition -> `data/followup/coherence/<cond>.json` |
| `followup_analyze.py` (new) | -> `data/followup/results.json` (does not touch `data/results.json`) |
| `run_followup.sh` (new) | Runs everything serially under memguard (11 GB, peak observed 9.8 GB) |

New data: `data/convs/` and `data/judged/` for ml_away, ml_random, dyn_gate, ml_away_raw, sl_matched, dyn_gate_raw and ml_random_raw, plus `data/followup/*`. No data from the original study was modified.

We used the existing even layers 12-28 (9 layers). The vectors already exist there, so no new readout was needed, and adjacent layers' directions are very similar (cosine 0.86-0.94), so adding the odd layers would mostly act like a larger budget. The Steer hook adds to the MLP output, i.e. the residual leaving that layer, which is the same convention the readout uses.

## Conditions

| condition | direction | layers | dose | total injected norm |
|---|---|---|---|---|
| cap_minus (prior) | raw capitulation | L20 | -0.4 x L20 norm | 4.5 |
| ml_away | cleaned capitulation | 12-28 | -0.5 budget | 6.7 |
| ml_random | random unit per layer | 12-28 | -0.5 budget | 6.7 |
| dyn_gate | cleaned (cap + syc) | 12-28 | gated: 0 / -0.25 / -0.5 | 0 to 6.7 |
| ml_away_raw | raw capitulation | 12-28 | -0.75 budget | 10.0 |
| sl_matched | raw capitulation | L20 | -0.883 x L20 norm | 10.0 |
| dyn_gate_raw | raw capitulation | 12-28 | gated: 0 / -0.375 / -0.75 | 0 to 10.0 |
| ml_random_raw | random unit per layer (same vectors as ml_random) | 12-28 | -0.75 budget | 10.0 |

## Common-component removal

Pairwise cosines of the unit trait vectors, before and after projecting out their mean direction:

| layer | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 |
|---|---|---|---|---|---|---|---|---|---|
| cap/syc before | +0.19 | +0.22 | +0.29 | +0.34 | +0.35 | +0.35 | +0.35 | +0.31 | +0.32 |
| cap/syc after | -0.61 | -0.61 | -0.59 | -0.61 | -0.58 | -0.58 | -0.58 | -0.59 | -0.59 |
| cap/enm before | +0.21 | +0.24 | +0.31 | +0.36 | +0.35 | +0.38 | +0.37 | +0.35 | +0.35 |
| cap/enm after | -0.59 | -0.59 | -0.58 | -0.58 | -0.58 | -0.55 | -0.56 | -0.54 | -0.55 |
| syc/enm before | +0.47 | +0.50 | +0.51 | +0.56 | +0.53 | +0.52 | +0.52 | +0.49 | +0.51 |
| syc/enm after | -0.29 | -0.28 | -0.31 | -0.30 | -0.34 | -0.36 | -0.35 | -0.36 | -0.35 |
| cleaned cap . raw cap | 0.77 | 0.75 | 0.71 | 0.69 | 0.68 | 0.67 | 0.68 | 0.69 | 0.69 |

With only three vectors, removing their mean necessarily leaves them pointing away from one another. Cosine of each away push with the raw trait directions at L20 (other layers are within 0.05):

| push | raw capitulation | raw sycophancy | raw enmeshment |
|---|---|---|---|
| ml_away = -cap_clean | -0.68 | +0.34 | +0.34 |
| dyn_gate = -(cap+syc)_clean | -0.31 | -0.27 | +0.59 |
| ml_away_raw, sl_matched, dyn_gate_raw = -cap_raw | -1.00 | -0.35 | -0.35 |
| ml_random | -0.01 | +0.02 | -0.01 |

## Calibration (pressured multi-turn states)

The 10 baseline transcripts were cut before assistant turns 4 and 6 (20 states, where baseline judged capitulation is already 57-70, so there is no floor effect). The next reply was generated at each dose (greedy, 220 tokens). Sonnet 5.5 scored capitulation, sycophancy and coherence with the earlier turns in view. The rule picks the strongest away dose whose coherence is within 5 points of unsteered. Each sweep rescored the same unsteered replies, giving 82.4, 82.3 and 82.8, so scorer noise is small.

| dose | capitulation | sycophancy | coherence | words |
|---|---|---|---|---|
| unsteered | 72.2-72.5 | 70.2-71.3 | 82.3-82.8 | 170 |
| cleaned, multi-layer -0.5 (chosen for ml_away) | 68.5 | 68.9 | 81.5 | 172 |
| cleaned, multi-layer -1.0 | 60.6 | 66.5 | 65.2 | 176 |
| cleaned, multi-layer -1.5 | 50.0 | 69.0 | 20.6 | 176 |
| cleaned, multi-layer -2.0 | 47.0 | 73.6 | 6.3 | 173 |
| cleaned, multi-layer -3.0 | 3.0 | 3.9 | 3.4 | 153 |
| cleaned cap+syc, multi-layer -0.5 (dyn_gate) | 68.1 | 66.3 | 75.7 | 170 |
| raw, multi-layer -0.5 | 53.0 | 56.9 | 80.5 | 179 |
| **raw, multi-layer -0.75 (chosen for ml_away_raw)** | 46.1 | 47.5 | 80.2 | 182 |
| raw, multi-layer -1.0 | 41.7 | 44.3 | 77.2 | 181 |
| **raw, single layer L20, same norm 10.0 (sl_matched)** | 50.9 | 53.2 | 77.0 | 182 |

At equal total norm, the single-layer push reduced capitulation slightly less on these 20 states (50.9 vs 46.1) and cost a little more coherence (5.8 vs 2.1 points). It did not collapse; its lowest single score was 60, against 62 for multi-layer. By the 5-point rule it was just outside, and it was run anyway as the layers-vs-dose control. The full conversations reversed the capitulation ordering (below) and widened the coherence gap.

Gate: the pre-reply sycophancy readout (L20, raw unit, last prompt token, steering off) over all 80 baseline turns. Quantiles were q10 -1.60, q25 -1.06, q50 -0.33, q75 +0.64, q90 +1.20. It correlates with the judged sycophancy of the reply that follows at Spearman 0.72 (0.67 after removing per-turn means). The ramp runs from 0 at q50 to the full dose at q90, snapped to 0 / half / full buckets.

## Results

Means over 10 conversations, with bootstrap 95% CIs. Trait scores are 0-100, averaged over both judges and all 8 turns. Markers are healthy-answer rates pooled over both judges. Coherence comes from `followup_coherence.py` (same two judges, mean of both).

| condition | capitulation | sycophancy | enmeshment | ETA | EH | RC | ABS | coherence | degenerate turns | words |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 57.4 [45.3, 67.5] | 66.4 [54.3, 76.8] | 27.1 [20.8, 32.3] | 0.855 | 0.507 | 0.801 | 0.688 | 86.7 [84.3, 88.9] | 0.0% | 169 |
| cap_minus (prior) | 54.8 [41.0, 66.4] | 62.2 [47.8, 75.4] | 27.8 [20.3, 35.2] | 0.900 | 0.568 | 0.793 | 0.701 | 84.9 [79.3, 89.2] | 2.5% | 175 |
| cap_plus (prior) | 67.7 [63.5, 71.7] | 70.7 | 31.1 | 0.693 | 0.481 | 0.793 | 0.668 | 75.6 [71.7, 78.6] | 8.8% | 150 |
| random (prior) | 58.8 [51.1, 66.3] | 62.0 [51.8, 71.7] | 29.8 [26.6, 32.7] | 0.894 | 0.518 | 0.795 | 0.665 | 87.2 [85.1, 89.2] | 0.0% | 170 |
| ml_away | 59.9 [48.7, 69.7] | 68.7 [55.4, 78.8] | 34.5 [28.1, 41.3] | 0.884 | 0.496 | 0.767 | 0.677 | 78.1 [73.1, 83.0] | 2.5% | 171 |
| ml_random | 60.0 [47.4, 70.3] | 65.5 [53.1, 74.5] | 30.6 [24.3, 35.3] | 0.876 | 0.534 | 0.798 | 0.699 | 88.3 [86.2, 90.2] | 0.0% | 169 |
| dyn_gate | 57.8 [42.2, 70.8] | 59.2 [44.8, 70.8] | 28.5 [23.0, 33.3] | 0.871 | 0.546 | 0.805 | 0.693 | 82.9 [79.3, 86.4] | 1.3% | 169 |
| ml_away_raw | 45.8 [30.1, 59.1] | 51.1 [39.1, 62.4] | 27.6 [22.6, 31.9] | 0.981 | 0.661 | 0.810 | 0.712 | 74.5 [67.4, 81.2] | 12.5% | 183 |
| sl_matched | 37.6 [22.6, 52.0] | 46.4 [33.4, 59.3] | 27.6 [23.3, 30.8] | 0.933 | 0.691 | 0.820 | 0.753 | 67.8 [58.0, 76.5] | 22.5% | 184 |
| dyn_gate_raw | 63.0 [57.1, 69.5] | 61.3 [51.7, 70.7] | 28.2 [22.5, 33.2] | 0.922 | 0.514 | 0.795 | 0.673 | 87.8 [86.7, 88.8] | 1.3% | 172 |
| ml_random_raw | 67.1 [57.6, 74.5] | 65.9 [51.7, 76.0] | 30.3 [22.5, 36.9] | 0.827 | 0.487 | 0.780 | 0.670 | 87.9 [85.0, 90.0] | 1.3% | 169 |

(cap_plus sycophancy and enmeshment CIs are in `data/results.json`; it is included here for the coherence comparison.)

### Capitulation separated from coherence

A turn counts as coherent if the two judges' mean coherence is at least 70 and neither flags it as degenerate. Incoherent turns cluster late in conversations, where baseline capitulation is highest. A plain "capitulation on coherent turns" mean therefore flatters the steered conditions. Each steered turn is instead compared with baseline at the same scenario and turn. The table gives the mean difference per conversation (negative = less capitulation), a bootstrap CI over the 10 conversations, and the sign test over scenarios.

| condition | coherent turns | coherent turns only, vs baseline | all turns, vs baseline | incoherent turns, vs baseline |
|---|---|---|---|---|
| cap_minus | 72/80 | +0.5 [-11.5, 12.2], 6/4, p 0.75 | -2.6 [-16.3, 11.6] | -27.2 (8 turns) |
| cap_plus | 63/80 | +9.9 [-1.3, 22.6], 3/7 | +10.3 [-0.7, 23.2] | +8.4 |
| random | 80/80 | +1.5 [-6.6, 10.0], 6/4 | +1.5 | none |
| ml_away | 69/80 | +2.2 [-7.7, 12.8], 4/6 | +2.5 | +2.7 |
| ml_random | 80/80 | +2.6 [-8.0, 12.7], 2/8 | +2.6 | none |
| dyn_gate | 73/80 | +2.1 [-9.7, 15.2], 5/5 | +0.4 | -25.8 |
| ml_away_raw | 57/80 | **-3.9 [-14.2, 9.4], 8/2, p 0.11** | -11.5 [-25.3, 3.5] | -35.2 (23 turns) |
| sl_matched | 51/80 | **-11.8 [-19.7, -4.9], 9/1, p 0.02** | -19.8 [-32.6, -7.1] | -35.4 (29 turns) |
| dyn_gate_raw | 79/80 | +5.7 [-4.7, 17.9], 5/5 | +5.7 | -2.5 (1 turn) |
| ml_random_raw | 79/80 | +10.4 [-0.6, 22.7], 3/7 | +9.8 | -60.0 (1 turn) |

Direct comparison of sl_matched with ml_away_raw (paired by scenario): capitulation 7/3 in favour of sl_matched (-8.2, p 0.34), coherence 3/7 (-6.7, p 0.34), coherent-turn capitulation 6/4 (-4.2, p 0.75). Neither is reliably better than the other at n = 10. The point estimates say the single layer is more effective and less coherent.

The judges' issue notes show what "degenerate" means here. Of the degenerate turns, 22 of 31 (sl_matched) and 15 of 16 (ml_away_raw) are described as verbatim or near-verbatim repeats of an earlier reply that ignore the user's new message. Other notes include self-contradiction ("refuses then states number") and factual slips. There are no word-salad loops. Replies with a word repeated four times in a row: 0 in every condition. A repeated earlier refusal scores as low capitulation, but it is not the model engaging and holding a position.

### Per-turn curves (mean of 10 conversations)

Judged capitulation:

| condition | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
|---|---|---|---|---|---|---|---|---|
| baseline | 16.8 | 38.2 | 50.0 | 57.0 | 67.2 | 69.5 | 76.9 | 83.3 |
| cap_minus | 19.2 | 33.5 | 43.0 | 56.0 | 65.1 | 73.0 | 74.0 | 74.2 |
| ml_away | 22.8 | 37.5 | 62.0 | 60.5 | 60.0 | 77.3 | 83.5 | 75.6 |
| ml_random | 20.2 | 43.0 | 56.9 | 56.5 | 70.0 | 72.8 | 82.0 | 78.5 |
| dyn_gate | 18.0 | 45.2 | 56.2 | 59.2 | 60.0 | 64.5 | 75.1 | 83.7 |
| ml_away_raw | 14.0 | 35.8 | 41.4 | 41.2 | 55.8 | 65.5 | 58.8 | 54.1 |
| sl_matched | 17.0 | 37.0 | 47.0 | 46.1 | 31.9 | 37.9 | 46.0 | 38.0 |
| dyn_gate_raw | 17.0 | 52.0 | 59.5 | 70.5 | 79.0 | 70.0 | 79.8 | 76.7 |
| ml_random_raw | 23.0 | 51.5 | 63.0 | 72.7 | 76.2 | 80.7 | 83.1 | 87.0 |

Judged capitulation, coherent turns only (not turn-matched; read alongside the coherence curve):

| condition | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
|---|---|---|---|---|---|---|---|---|
| baseline | 16.8 | 38.2 | 50.0 | 57.0 | 67.2 | 69.5 | 76.9 | 83.3 |
| ml_away_raw | 14.0 | 33.1 | 39.1 | 55.0 | 77.5 | 82.2 | 80.5 | 80.8 |
| sl_matched | 17.0 | 40.8 | 63.3 | 62.3 | 43.1 | 54.5 | 62.0 | 49.7 |

ml_away_raw's late-turn gap (54 vs 83 at turn 8) is entirely on incoherent turns. Its coherent late turns capitulate as much as baseline does. sl_matched keeps a gap on coherent late turns.

Coherence (full conversations):

| condition | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
|---|---|---|---|---|---|---|---|---|
| baseline | 90.0 | 89.5 | 88.8 | 85.8 | 85.7 | 83.8 | 86.2 | 83.8 |
| cap_minus | 92.2 | 88.2 | 86.9 | 85.3 | 85.0 | 82.5 | 75.8 | 83.3 |
| cap_plus | 86.2 | 81.5 | 73.8 | 72.8 | 80.0 | 74.8 | 71.0 | 64.5 |
| ml_away | 82.5 | 81.7 | 79.8 | 81.1 | 80.2 | 78.5 | 70.3 | 70.7 |
| ml_random | 90.1 | 89.5 | 88.6 | 87.9 | 88.8 | 87.2 | 87.0 | 87.3 |
| dyn_gate | 89.2 | 86.3 | 86.3 | 82.9 | 83.6 | 81.2 | 80.4 | 73.3 |
| ml_away_raw | 86.2 | 82.2 | 78.7 | 74.2 | 70.2 | 72.9 | 64.9 | 66.8 |
| sl_matched | 85.0 | 77.8 | 72.2 | 64.7 | 66.2 | 59.0 | 49.0 | 68.2 |
| dyn_gate_raw | 92.1 | 89.8 | 89.0 | 89.0 | 88.3 | 86.4 | 84.3 | 83.8 |
| ml_random_raw | 92.0 | 87.3 | 87.2 | 89.2 | 87.1 | 88.1 | 86.4 | 85.5 |

Coherence decays with turn number under a constant push. The first reply is close to normal, and the damage accumulates as steered replies fill the context.

Sycophancy:

| condition | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
|---|---|---|---|---|---|---|---|---|
| baseline | 43 | 60 | 68 | 66 | 71 | 72 | 74 | 79 |
| ml_away | 48 | 57 | 73 | 68 | 68 | 80 | 82 | 75 |
| dyn_gate | 31 | 52 | 59 | 60 | 60 | 62 | 74 | 76 |
| ml_away_raw | 29 | 45 | 48 | 49 | 58 | 62 | 61 | 56 |
| sl_matched | 32 | 50 | 56 | 55 | 40 | 45 | 51 | 42 |
| dyn_gate_raw | 37 | 58 | 60 | 68 | 69 | 62 | 70 | 66 |

### Paired per-scenario comparisons

Improve / worse out of 10 (ties excluded), mean difference, two-sided sign test p, Wilcoxon p. Lower capitulation, sycophancy or enmeshment counts as improving; higher coherence or a higher marker rate counts as improving. The full set, including every pair, is in `data/followup/results.json` under `paired`.

| comparison | capitulation | sycophancy | enmeshment | EH | coherence |
|---|---|---|---|---|---|
| cap_minus vs baseline | 6/4, -2.6, p 0.75 | 5/5, -4.2 | 4/6, +0.7 | 5/5 | 5/5, -1.8 |
| ml_away vs baseline | 4/6, +2.5, p 0.75 | 5/5, +2.4 | 3/7, +7.5, p 0.34 | 5/5 | 1/9, -8.6, p 0.02 |
| ml_away vs ml_random | 4/6, -0.1 | 3/6, +3.3 | 4/6, +4.0 | 2/7 | 1/9, -10.2, p 0.02 |
| ml_random vs baseline | 2/8, +2.6, p 0.11 | 5/5, -0.9 | 4/6, +3.5 | 6/3 | 6/4, +1.6 |
| dyn_gate vs baseline | 5/5, +0.4 | 6/4, -7.2 (w 0.13) | 5/5, +1.4 | 5/5 | 2/8, -3.8 |
| ml_away_raw vs baseline | 9/1, -11.5, p 0.02 | 10/0, -15.3, p 0.002 | 7/3, +0.5 | 9/1, p 0.02 | 0/10, -12.2, p 0.002 |
| ml_away_raw vs ml_random | 8/2, -14.2, p 0.11 | 9/1, -14.4, p 0.02 | 6/4, -3.0 | 9/1, p 0.02 | 0/10, -13.8, p 0.002 |
| ml_away_raw vs ml_away | 8/2, -14.1, p 0.11 | 9/1, -17.6, p 0.02 | 8/2, -7.0, p 0.11 | 9/1, p 0.02 | 5/5, -3.6 |
| sl_matched vs baseline | 8/2, -19.8, p 0.11 (w 0.02) | 9/1, -20.0, p 0.02 | 6/3, +0.6 | 8/1, p 0.04 | 0/10, -18.9, p 0.002 |
| sl_matched vs cap_minus | 8/2, -17.2, p 0.11 | 9/1, -15.8, p 0.02 | 6/4, -0.2 | 8/2, p 0.11 | 0/10, -17.1, p 0.002 |
| sl_matched vs ml_away_raw | 7/3, -8.2, p 0.34 | 6/4, -4.7, p 0.75 | 6/4, +0.1 | 4/6 | 3/7, -6.7, p 0.34 |
| dyn_gate_raw vs baseline | 5/5, +5.7, p 1.0 | 7/3, -5.1, p 0.34 | 5/5, +1.1 | 7/3 | 4/6, +1.2 |
| dyn_gate_raw vs ml_away_raw | 3/7, +17.2, p 0.34 | 3/7, +10.2 | 5/4, +0.6 | 2/8, p 0.11 | 9/1, +13.4, p 0.02 |
| ml_random_raw vs baseline | 3/7, +9.8, p 0.34 | 5/5, -0.5 | | 6/4 | 7/3, +1.2 |
| ml_away_raw vs ml_random_raw | 10/0, -21.3, p 0.002 | 9/1, -14.8, p 0.02 | | 9/1, p 0.02 | 0/10, -13.4, p 0.002 |
| sl_matched vs ml_random_raw | 9/1, -29.5, p 0.02 | 8/2, -19.5, p 0.11 | | 8/2 | 0/10, -20.1, p 0.002 |

With n = 10, even 8/2 gives p = 0.11. Nothing here would survive a correction for the number of comparisons.

### Internal check (L20 response-mean readout, mean projection on the raw unit trait directions)

| condition | capitulation | sycophancy | enmeshment |
|---|---|---|---|
| baseline | -1.14 | +0.72 | -0.50 |
| ml_random | -1.36 | +0.76 | -0.50 |
| ml_away (cleaned) | -2.95 | +2.11 | +1.14 |
| dyn_gate | -1.26 | +0.78 | -0.28 |
| ml_away_raw | -5.68 | -0.52 | -1.77 |
| sl_matched (L20 is its push layer, so this includes the full -10.0 push) | -12.09 | -2.85 | -4.12 |

The random multi-layer push leaves all three readouts where they were. The cleaned push moves the model away from capitulation but towards sycophancy (+1.4) and enmeshment (+1.6), by more than the injected vector alone. The judges' scores are weaker here: enmeshment rose 7.5 under ml_away but 3.5 under ml_random too, so about half of the judged rise is shared with a random push. The raw pushes also lower the sycophancy and enmeshment readouts, about as the 0.35 overlap of the raw capitulation vector predicts. Part of their sycophancy reduction therefore comes from the direction's shared component by construction. Judged enmeshment did not move.

### Gate behaviour

| | dyn_gate (cleaned cap+syc, full -0.5) | dyn_gate_raw (raw cap, full -0.75) |
|---|---|---|
| turns unsteered / half / full | 42 / 26 / 12 | 50 / 27 / 3 |
| share steered per turn T1..T8 | 0.2 0.4 0.4 0.7 0.3 0.5 0.6 0.7 | 0.3 0.2 0.5 0.4 0.6 0.4 0.4 0.2 |
| mean dose per turn T1..T8 | -0.08 -0.15 -0.13 -0.20 -0.08 -0.18 -0.23 -0.23 | -0.15 -0.08 -0.23 -0.19 -0.23 -0.15 -0.15 -0.08 |

The gate reads the state with steering off, but the context contains earlier steered replies. Under dyn_gate_raw the pre-reply sycophancy readings stayed lower (per-turn means -0.72 to -0.14, against baseline per-turn means up to +0.27). This is consistent with the raw push lowering sycophancy and so switching its own gate off, though the conversations differ, so it is not a controlled comparison. In practice the gate also selects scenarios more than turns. Under dyn_gate one relationship scenario was steered on every turn, and two finance and healthcare scenarios were never steered. The largest dyn_gate capitulation change in either direction was in a never-steered conversation (-45), which is a direct measure of run-to-run tester variance.

## Interpretation

The single-layer null result in the original study was a dose problem, not a layer problem. That calibration used single-turn prompts where capitulation was already at the floor, so it could not see any benefit, only cost, and its coherence rule capped the push at a size too small to matter. Calibrated on mid-conversation states, larger pushes along the same raw direction do reduce judged capitulation, at one layer as well as across nine. At equal total norm the single layer moved capitulation further (37.6 vs 45.8), while the nine-layer version damaged the text less (coherence 74.5 vs 67.8). Spreading the push across layers buys some coherence per unit of push; it does not buy effectiveness.

None of the pushes that changed behaviour did so for free. Both raw pushes degraded the full conversations progressively. From turn 4 onwards, 4 or 5 of the 10 conversations under ml_away_raw, and 3 to 7 under sl_matched, were judged incoherent at each turn, usually because the model repeated an earlier reply instead of answering. Those degenerate turns account for much of the measured improvement. Restricted to coherent turns and compared turn for turn with baseline, the multi-layer effect is small and uncertain (-3.9). The single-layer effect survives at the 60 and 70 cut-offs (-14.7 and -11.8, 9 of 10 scenarios each) but not at 80, where too few of its turns remain to say. So at this dose part of the change may be a real shift in how the model answers, and a large part is the model losing the thread. On coherent turns the multi-layer push still lowers sycophancy and improves the honesty and attunement markers, which suggests its clearest effect on fluent replies is on flattery and candour rather than on giving in.

Our calibration missed this because it scored one reply from a baseline context. Degradation under steering compounds over turns as steered replies accumulate in the context, so single-reply coherence overstates what the dose will do across eight turns. A calibration rule for multi-turn steering should score coherence at the end of a steered conversation, not after one reply.

Common-component removal with three vectors hurt, for the geometric reason in the table above. Per-turn gating is the one approach that kept coherence intact (87.8), but our gate pushed too rarely and too softly to change capitulation. Its own steering lowered the signal it was gating on. A gate that switches on earlier (say from q25) or uses a full dose when it fires is the obvious next test. It would aim to recover the coherent-turn effect seen with sl_matched without the progressive degradation.

## Caveats

- n = 10 conversations per condition. Bootstrap CIs on mean capitulation span 12-30 points, and one scenario can move by 45 points between runs with no intervention. Nothing here survives correction for the number of comparisons.
- Coherence was scored by the same Claude judges that score capitulation, after the fact and with condition hidden. The threshold of 70 is ours. Coherent-turn analyses drop different numbers of turns per condition (51 to 80 of 80).
- The raw direction overlaps sycophancy and enmeshment (cosine 0.35). The sycophancy drops are partly by construction, and "capitulation-specific" steering is not what was shown.
- Steered replies under ml_away_raw and sl_matched were about 8% longer. Longer replies have more room for caveats, which the rubric rewards.
- The tester adapts to the target, so conditions diverge from turn 2. Paired-by-scenario comparisons are paired on the setup, not the dialogue.
- dyn_gate and dyn_gate_raw doses were not separately calibrated.
- One rubric judgement failed for baseline, cap_minus and ml_away (one judge on one conversation each). Those conversations use the other judge only. Bootstrap CIs differ by up to 1 point from earlier drafts because the resampling order changed.
- All labels come from Claude models (tester, calibration scorer, rubric and coherence judges).
- re:dvx's own implementation could not be consulted, so "re:dvx-style" here means our reading of the ideas, not its code.
