# Shared helpers for the mac-verify scripts. Every step is recorded in $OUT/summary.md as
# PASS / FAIL / NOT RUN (+ reason); NOT RUN is never counted as a pass.
OUT="${OUT:?OUT must be set}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.md"
FAILS=0
[ -f "$SUMMARY" ] || printf '| step | result | log |\n|---|---|---|\n' > "$SUMMARY"

# run <label> <cmd...> : runs the command, tees to $OUT/<label>.log, records PASS/FAIL.
run() {
  local label="$1"; shift
  local log="$OUT/$label.log"
  echo "::group::$label"
  echo "+ $*" | tee "$log"
  set +e
  "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "::endgroup::"
  if [ "$rc" -eq 0 ]; then
    echo "| $label | PASS | $label.log |" >> "$SUMMARY"
  else
    echo "| $label | FAIL (exit $rc) | $label.log |" >> "$SUMMARY"
    echo "::error::$label failed (exit $rc)"
    annotate_log "$label" "$log"
    FAILS=$((FAILS + 1))
  fi
  return 0
}

# annotate_log <label> <log>: surface the important lines of a failed step as annotations
# (readable through the checks API even where artifact / log downloads are blocked).
annotate_log() {
  local label="$1" log="$2" n=0
  while IFS= read -r line && [ $n -lt 8 ]; do
    line="${line//'%'/'%25'}"
    echo "::error title=$label::${line:0:900}"
    n=$((n + 1))
  done < <(grep -E -A3 "^error(\[|:)|panicked at|FAILED|^failures:|assertion" "$log" | grep -vE "^--$" | head -8)
  # last lines too (build scripts often end with the real cause)
  echo "::error title=$label (tail)::$(tail -n 12 "$log" | sed 's/$/ |/' | tr -d '\n' | cut -c1-900)"
}

# not_run <label> <reason>
not_run() {
  echo "| $1 | NOT RUN — $2 | |" >> "$SUMMARY"
  echo "::warning::$1 not run: $2"
}

# require_measured <run-label> <pattern> : a run that "passed" by skipping itself (no measurement line in its
# log) is rewritten from PASS to NOT RUN in the summary — a skipped run never counts as a pass.
require_measured() {
  local label="$1" pat="$2" log="$OUT/$1.log"
  if ! grep -qE "$pat" "$log" 2>/dev/null; then
    sed -i.bak -E "s/^\| $label \| PASS \|/| $label | NOT RUN — test skipped itself (no \/$pat\/ line) |/" "$SUMMARY" && rm -f "$SUMMARY.bak"
    echo "::warning::$label: test produced no measurement (skipped)"
  fi
}

# Measurement lines and the summary table as a few long notices (annotations are readable through the
# checks API; artifact / log blobs may not be reachable from where the results are read).
publish_notices() {
  local tmp="$OUT/.measure.txt"
  grep -hE "\[(GPU-EQ|DECISION-EQ|EMBED-BATCHED-TEST|CPU-PACKED|EMBED-CPU-PACKED|[A-Za-z-]+-M-INV|[A-Za-z-]+-B-INV|UTF8-STREAM|CPU-3-2)\]|QWEN4-EXP\] +variant|DEVICE_PROBE|^- |^test result" "$OUT"/*.log 2>/dev/null | grep -v '^+ ' \
    | cut -c1-600 > "$tmp" || true
  echo "::notice title=summary::$(grep '^| ' "$SUMMARY" | tail -n +2 | sed 's/$/ ;;/' | tr -d '\n' | cut -c1-3900)"
  local chunk="" n=0
  while IFS= read -r line; do
    if [ $(( ${#chunk} + ${#line} )) -gt 3800 ]; then
      echo "::notice title=measure-$n::$chunk"; n=$((n + 1)); chunk=""
      [ $n -ge 8 ] && break
    fi
    chunk="$chunk$line ;; "
  done < "$tmp"
  [ -n "$chunk" ] && [ $n -lt 8 ] && echo "::notice title=measure-$n::$chunk"
  return 0
}

finish() {
  publish_notices
  echo; cat "$SUMMARY"
  cat "$SUMMARY" >> "${GITHUB_STEP_SUMMARY:-/dev/null}" || true
  [ "$FAILS" -eq 0 ]
}
