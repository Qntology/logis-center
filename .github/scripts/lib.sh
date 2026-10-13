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
    FAILS=$((FAILS + 1))
  fi
  return 0
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

finish() {
  echo; cat "$SUMMARY"
  cat "$SUMMARY" >> "${GITHUB_STEP_SUMMARY:-/dev/null}" || true
  [ "$FAILS" -eq 0 ]
}
