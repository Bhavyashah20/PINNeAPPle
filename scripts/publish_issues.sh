#!/usr/bin/env bash
# Publish docs/dev/issues/*.md as GitHub issues (needs `gh auth login`).
# Idempotent: skips files whose exact title already exists as an issue.
# Usage: scripts/publish_issues.sh [owner/repo]   (default: PINNeAPPle-Labs/PINNeAPPle)
set -euo pipefail
REPO="${1:-PINNeAPPle-Labs/PINNeAPPle}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/docs/dev/issues"
existing="$(gh issue list --repo "$REPO" --state all --limit 1000 --json title --jq '.[].title')"
for f in "$DIR"/[0-9][0-9][0-9]-*.md; do
  title="$(sed -n 's/^title: "\(.*\)"$/\1/p' "$f" | head -1)"
  labels="$(sed -n 's/^labels: \[\(.*\)\]$/\1/p' "$f" | head -1 | tr -d ' ')"
  if grep -Fxq -- "$title" <<<"$existing"; then echo "skip: $title"; continue; fi
  body="$(awk 'BEGIN{n=0} /^---$/{n++; next} n>=2' "$f")"
  IFS=',' read -ra L <<<"$labels"
  for l in "${L[@]}"; do gh label create "$l" --repo "$REPO" >/dev/null 2>&1 || true; done
  gh issue create --repo "$REPO" --title "$title" --body "$body" --label "$labels"
done
