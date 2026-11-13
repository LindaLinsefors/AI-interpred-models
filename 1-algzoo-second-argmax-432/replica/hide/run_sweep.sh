#!/bin/sh
cd "$(dirname "$0")"
for bs in 64 128 256 512 1000; do for seed in 0 1 2; do
  python -B train_replica.py --batch_size $bs --seed $seed --out runs/bs${bs}_s${seed} > runs/bs${bs}_s${seed}.log 2>&1 &
done; done
wait
