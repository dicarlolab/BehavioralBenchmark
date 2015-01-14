#!/bin/bash
python make_tunnel.py
python feature_subsample.py $1 $2
