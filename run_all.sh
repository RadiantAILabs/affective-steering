#!/bin/bash
# The whole pipeline, one model in memory at a time, each GPU job under the memory watchdog (memguard.py).
# MAX_GB=11 suits a 16 GB Mac; raise it on bigger machines. Pass your own scenarios/prompts/rubric through the
# environment (SCENARIOS_DIR, TESTER_PROMPT, AGENT_PROMPT, RUBRIC); the defaults are the examples in examples/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
MAX_GB=${MAX_GB:-11}
G=($PY memguard.py --max-gb "$MAX_GB" --)
CONV_ARGS=(--scenarios-dir "${SCENARIOS_DIR:-examples/scenarios}" --tester-prompt "${TESTER_PROMPT:-examples/tester_prompt.example.md}"
           --agent-prompt "${AGENT_PROMPT:-examples/agent_prompt.example.md}")
mkdir -p logs

"${G[@]}" $PY persona.py > logs/persona.log 2>&1
"${G[@]}" $PY converse.py --condition baseline "${CONV_ARGS[@]}" > logs/conv_baseline.log 2>&1
"${G[@]}" $PY calib.py > logs/calib.log 2>&1
MINUS=$($PY -c "import json;print(json.load(open('data/calib.json'))['chosen']['minus'])")
PLUS=$($PY -c "import json;print(json.load(open('data/calib.json'))['chosen']['plus'])")
"${G[@]}" $PY converse.py --condition cap_minus --trait capitulation --coef "$MINUS" "${CONV_ARGS[@]}" > logs/conv_cap_minus.log 2>&1
"${G[@]}" $PY converse.py --condition random --random-like capitulation --coef "$MINUS" "${CONV_ARGS[@]}" > logs/conv_random.log 2>&1
"${G[@]}" $PY converse.py --condition cap_plus --trait capitulation --coef "$PLUS" "${CONV_ARGS[@]}" > logs/conv_cap_plus.log 2>&1
"${G[@]}" $PY persona_space.py > logs/persona_space.log 2>&1
# Claude targets need no local model
$PY converse.py --condition haiku --api-target haiku "${CONV_ARGS[@]}" > logs/conv_haiku.log 2>&1
$PY converse.py --condition sonnet --api-target sonnet "${CONV_ARGS[@]}" > logs/conv_sonnet.log 2>&1
$PY judge.py --rubric "${RUBRIC:-examples/rubric.example.json}" baseline cap_minus random cap_plus haiku sonnet > logs/judge.log 2>&1
$PY analyze.py --rubric "${RUBRIC:-examples/rubric.example.json}" > logs/analyze.log 2>&1
echo "done"
