#!/usr/bin/env bash
# Free the page cache before the server loads and after it stops. On a unified-memory board (Jetson AGX Thor,
# DGX Spark) the cache sits in the memory the GPU allocates from, and TensorFold budgets only what is free at start:
# after the ~106 GiB checkpoint has been read, the cache can hold most of it. Needs root (start.sh and stop.sh run it
# with sudo). It only flushes and drops clean caches; nothing else on the system is changed.
set -euo pipefail
sync
echo 3 > /proc/sys/vm/drop_caches
