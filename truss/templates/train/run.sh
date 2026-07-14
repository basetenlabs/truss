#!/bin/bash
set -eux

pip install -r requirements.txt

python train.py
