#!/bin/bash
#$ -S /bin/bash


export PATH

source /home/babel/BABEL_OP3_404/releaseB/exp-graphemic-obr22/MLenv/bin/activate

year="$(date +'%Y')"
month="$(date +'%m')"
day="$(date +'%d')"
log_file="LOG/${year}_${month}_${day}_out.txt"


# Training:
OMP_NUM_THREADS=1 python main.py --nEpochs 25 --shuffle --dataset dev-grapheme-mapped-one-cn --target target_overlap_0.1 --lattice-type word --encoder TRANSFORMER --transformer_attn sdp --test_epochs --suffix temp --rootDir ../data --onebest

#OMP_NUM_THREADS=1 python main.py --batchSize 32 --nEpochs 25 --LR 0.01 --momentum 0.05 --shuffle --LRDecay newbob --dataset dev-grapheme-mapped-one-cn --target target_overlap_0.1 --lattice-type word --arc_combine-method attention --grapheme-combination concat --encoder-type ATTENTION --attention_order all --attention_key global --intermediate_dropout 0 --attention_heads 1 --test_epochs --suffix global_test_2 --clip 10 --rootDir ../data --onebest

#OMP_NUM_THREADS=1 python main.py --batchSize 32 --nEpochs 25 --LR 0.01 --momentum 0.05 --shuffle --LRDecay newbob --dataset dev-grapheme-mapped-one-cn --target target_overlap_0.1 --lattice-type word --arc_combine-method attention --grapheme-combination concat --encoder-type ATTENTION --attention_order all --attention_key global --intermediate_dropout 0 --attention_heads 1 --test_epochs --suffix global_test_3 --clip 10 --rootDir ../data --onebest

#OMP_NUM_THREADS=1 python main.py --batchSize 32 --nEpochs 25 --LR 0.01 --momentum 0.05 --shuffle --LRDecay newbob --dataset dev-grapheme-mapped-one-cn --target target_overlap_0.1 --lattice-type word --arc_combine-method attention --grapheme-combination concat --encoder-type ATTENTION --attention_order all --attention_key dist --intermediate_dropout 0 --attention_heads 1 --test_epochs --suffix test_4 --clip 10 --rootDir ../data --onebest

# Evaluation:
#OMP_NUM_THREADS=1 python main.py --batchSize 32 --nEpochs 25 --LR 0.01 --momentum 0.05 --shuffle --LRDecay newbob --dataset dev-grapheme-mapped-one-cn --target target_overlap_0.1 --lattice-type word --arc_combine-method attention --grapheme-combination concat --encoder-type ATTENTION --attention_order all --attention_key global --intermediate_dropout 0 --attention_heads 1 --suffix global_test_1 --clip 10 --rootDir ../data --onebest --epochNum -2 --testOnly --attention_stats --seq_length_stats

#OMP_NUM_THREADS=1 python main.py --batchSize 32 --nEpochs 25 --LR 0.01 --momentum 0.05 --shuffle --LRDecay newbob --dataset dev-grapheme-mapped-one-cn --target target_overlap_0.1 --lattice-type word --arc_combine-method attention --grapheme-combination concat --encoder-type ATTENTION --attention_order all --attention_key global --intermediate_dropout 0 --attention_heads 1 --suffix global_test_2 --clip 10 --rootDir ../data --onebest --epochNum -2 --testOnly --attention_stats --seq_length_stats

#OMP_NUM_THREADS=1 python main.py --batchSize 32 --nEpochs 25 --LR 0.01 --momentum 0.05 --shuffle --LRDecay newbob --dataset dev-grapheme-mapped-one-cn --target target_overlap_0.1 --lattice-type word --arc_combine-method attention --grapheme-combination concat --encoder-type ATTENTION --attention_order all --attention_key global --intermediate_dropout 0 --attention_heads 1 --suffix global_test_3 --clip 10 --rootDir ../data --onebest --epochNum -2 --testOnly --attention_stats --seq_length_stats

