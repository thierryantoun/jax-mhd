#!/bin/bash
set -e
MODE=${1:-cpu}
IMPL=${2:-scan_concat}
case "$IMPL" in
    while)       MAIN_DIR="article/2nd_order_while" ;;
    scan)        MAIN_DIR="article/2nd_order_scan" ;;
    scan_concat) MAIN_DIR="article/2nd_order_scan_concat" ;;
    *) echo "Unknown implementation: $IMPL. Choose: while, scan, scan_concat"; exit 1 ;;
esac
MAIN="$MAIN_DIR/main.py"
INI="$MAIN_DIR/orszag-tang.ini"
mkdir -p results

if [ "$MODE" = "cpu" ] || [ "$MODE" = "all" ]; then
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python3 "$MAIN" "$INI" | tee results/${IMPL}_cpu.txt
fi
if [ "$MODE" = "gpu" ] || [ "$MODE" = "all" ]; then
    python3 "$MAIN" "$INI" | tee results/${IMPL}_gpu.txt
fi
