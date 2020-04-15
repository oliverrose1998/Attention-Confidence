#!/bin/bash
#$ -S /bin/bash


export PATH

source /home/babel/BABEL_OP3_404/releaseB/exp-graphemic-obr22/MLenv/bin/activate

year="$(date +'%Y')"
month="$(date +'%m')"
day="$(date +'%d')"
log_file="LOG/${year}_${month}_${day}_out.txt"


# Training:
OMP_NUM_THREADS=1 python main.py --nEpochs 25 --shuffle --dataset dev-grapheme-mapped-one-cn --target target_overlap_0.1 --lattice-type word --encoder TRANSFORMER --attn_type mult --test_epochs --suffix temp --rootDir ../data --onebest


OMP_NUM_THREADS=1 python main.py --nEpochs 25 --shuffle --dataset dev-grapheme-mapped-one-cn --target target_overlap_0.1 --lattice-type word --encoder TRANSFORMER --attn_type mult --test_epochs --suffix temp --rootDir ../data --onebest --epochNum -2 --testOnly --attention_stats --seq_length_stats
