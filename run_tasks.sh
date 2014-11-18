#!/bin/sh
sbatch -n 11 --mem=250000 run.sh basic.py $1
sbatch -n 11 --mem=250000 run.sh subordinate.py $1
