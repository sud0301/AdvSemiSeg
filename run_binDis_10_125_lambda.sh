#!/bin/bash

#PBS -l nodes=1:ppn=8:gpus=1:nvidiaTITANX
#PBS -l mem=12gb
#PBS -l walltime=24:00:00
#PBS -e myprog.err
#PBS -o myprog.out
#PBS -q default-cpu
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/dense_prediction/forked/AdvSemiSeg/
python train_binaryDis_10_lambda.py --snapshot-dir snapshots/binaryDis_10_125_sep  --partial-data 0.125  --num-steps 40000 >> out_binaryDis_10_125_sep.txt 
