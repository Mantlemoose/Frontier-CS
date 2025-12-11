#!/usr/bin/env bash
set -euo pipefail
# Start go-judge (sandbox)
/usr/local/bin/go-judge -parallelism "${GJ_PARALLELISM}" &
sleep 0.5
# Start orchestrator
node /app/server.js
