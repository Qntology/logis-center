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
    annotate_log "$label (exit $rc)" "$log"
    FAILS=$((FAILS + 1))
  fi
  return 0
}

# annotate_log <label> <log>: surface the important lines of a failed step as annotations
# (readable through the checks API even where artifact / log downloads are blocked).
annotate_log() {
  # One annotation per failed step (GitHub keeps only 10 error annotations per step): the error /
  # panic lines with their next lines, then the log tail, joined with " | ".
  local label="$1" log="$2"
  local body
  body="$(grep -E -A4 "^error(\[|:)|panicked at|^failures:|assertion|FAILED" "$log" | grep -vE "^--$|^\s*[0-9]+: |^\s+at " | head -40 | sed 's/$/ |/' | tr -d '\n' | cut -c1-2800)"
  echo "::error title=$label::$body || tail: $(tail -n 8 "$log" | sed 's/$/ |/' | tr -d '\n' | cut -c1-1000)"
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
  # Key lines first (one-line summaries), each tagged with its run label, then the detail lines.
  local tmp="$OUT/.measure.txt"
  local key="\[(EMBED-SUMMARY|DECISION-EQ)\]|\[KV-IO\] (write|best|cold read std)|QWEN4-EXP\] +variant_. on|-M-INV\] 측정 결과|-M-INV\] arch|\[CPU-3-2\]|DEVICE_PROBE|^test result: FAILED"
  local detail="\[(GPU-EQ|UTF8-STREAM|KV-IO)\]|^- |^test result"
  : > "$tmp"
  for f in "$OUT"/*.log; do
    [ -f "$f" ] || continue
    local tag; tag="$(basename "$f" .log)"
    grep -E "$key" "$f" 2>/dev/null | grep -v '^+ ' | cut -c1-500 | sed "s|^|<$tag> |" >> "$tmp" || true
  done
  for f in "$OUT"/*.log; do
    [ -f "$f" ] || continue
    local tag; tag="$(basename "$f" .log)"
    grep -E "$detail" "$f" 2>/dev/null | grep -vE "$key" | grep -v '^+ ' | cut -c1-300 | sed "s|^|<$tag> |" >> "$tmp" || true
  done
  echo "::notice title=summary::$(grep '^| ' "$SUMMARY" | tail -n +2 | sed 's/$/ ;;/' | tr -d '\n' | cut -c1-1800)"
  # Annotation messages are cut around 4 KB: keep each chunk well below that (Korean is 3 bytes/char).
  local chunk="" n=0
  while IFS= read -r line; do
    if [ $(( ${#chunk} + ${#line} )) -gt 1100 ] && [ -n "$chunk" ]; then
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
