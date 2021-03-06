#!/bin/bash

#PBS -l nodes=1:ppn=4:gpus=1:nvidiaTITANX
#PBS -l mem=12gb
#PBS -l walltime=24:00:00
#PBS -e myprog.err
#PBS -o myprog.out
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/dense_prediction/forked/AdvSemiSeg/
python train_binaryDis_10.py --snapshot-dir snapshots/binaryDis_10_50_2  --restore-from snapshots/binaryDis_10_50_2/VOC_20000.pth --restore-from-D snapshots/binaryDis_10_50_2/VOC_20000_D.pth --partial-data 0.50  --num-steps 40000  >> out_binaryDis_10_50_2.txt
