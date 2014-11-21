#!/bin/sh
echo $1
sbatch -n 11 --mem=250000 run.sh run_consistency.py $1 basic
sbatch -n 11 --mem=250000 run.sh run_consistency.py $1 subordinate
sbatch -n 11 --mem=250000 run.sh run_consistency.py $1 all
