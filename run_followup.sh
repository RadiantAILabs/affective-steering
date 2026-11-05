#!/bin/bash
# The follow-up (multi-layer, cleaned and gated steering, plus the coherence judge), one model in memory at a time,
# each GPU job under the memory watchdog (memguard.py). Run ./run_all.sh first: this needs its persona vectors, the
# baseline transcripts, data/calib.json and data/results.json. Same environment variables as run_all.sh.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
MAX_GB=${MAX_GB:-11}
G=($PY memguard.py --max-gb "$MAX_GB" --)
CONV_ARGS=(--scenarios-dir "${SCENARIOS_DIR:-examples/scenarios}" --tester-prompt "${TESTER_PROMPT:-examples/tester_prompt.example.md}"
           --agent-prompt "${AGENT_PROMPT:-examples/agent_prompt.example.md}")
RUBRIC=${RUBRIC:-examples/rubric.example.json}
mkdir -p logs data/followup/coherence

# directions: cleaned and raw trait units at layers 12-28, and a random control -> data/followup/ml_vectors.npz
$PY followup_vectors.py > logs/followup_vectors.log 2>&1
# gate quantiles + dose sweep for the cleaned direction, then the raw direction, then the norm-matched single layer
"${G[@]}" $PY followup_calib.py > logs/followup_calib.log 2>&1
CLEAN=$($PY -c "import json;print(json.load(open('data/followup/calib.json'))['chosen'])")
"${G[@]}" $PY followup_calib_raw.py > logs/followup_calib_raw.log 2>&1
RAW=$($PY -c "import json;print(json.load(open('data/followup/calib_raw.json'))['chosen'])")
"${G[@]}" $PY followup_calib_sl.py > logs/followup_calib_sl.log 2>&1
SL=$($PY -c "import json;print(round(json.load(open('data/followup/calib_sl.json'))['coef'],4))")

"${G[@]}" $PY converse.py --condition ml_away --ml cap_clean --coef "$CLEAN" "${CONV_ARGS[@]}" > logs/conv_ml_away.log 2>&1
"${G[@]}" $PY converse.py --condition ml_random --ml random --coef "$CLEAN" "${CONV_ARGS[@]}" > logs/conv_ml_random.log 2>&1
"${G[@]}" $PY converse.py --condition dyn_gate --ml capsyc_clean --coef "$CLEAN" --gate data/followup/gate.json "${CONV_ARGS[@]}" > logs/conv_dyn_gate.log 2>&1
"${G[@]}" $PY converse.py --condition ml_away_raw --ml capitulation_raw --coef "$RAW" "${CONV_ARGS[@]}" > logs/conv_ml_away_raw.log 2>&1
"${G[@]}" $PY converse.py --condition sl_matched --trait capitulation --layer 20 --coef "$SL" "${CONV_ARGS[@]}" > logs/conv_sl_matched.log 2>&1
"${G[@]}" $PY converse.py --condition dyn_gate_raw --ml capitulation_raw --coef "$RAW" --gate data/followup/gate.json "${CONV_ARGS[@]}" > logs/conv_dyn_gate_raw.log 2>&1
"${G[@]}" $PY converse.py --condition ml_random_raw --ml random --coef "$RAW" "${CONV_ARGS[@]}" > logs/conv_ml_random_raw.log 2>&1

NEW=(ml_away ml_random dyn_gate ml_away_raw sl_matched dyn_gate_raw ml_random_raw)
$PY judge.py --rubric "$RUBRIC" "${NEW[@]}" > logs/judge_followup.log 2>&1
$PY followup_coherence.py baseline cap_minus cap_plus random "${NEW[@]}" > logs/followup_coherence.log 2>&1
$PY followup_analyze.py --rubric "$RUBRIC" > logs/followup_analyze.log 2>&1
echo "done"
