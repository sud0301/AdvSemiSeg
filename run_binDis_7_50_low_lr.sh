#!/bin/bash

#PBS -l hostlist=^chip,nodes=1:ppn=8:gpus=1:nvidiaTITANX
#PBS -l mem=12gb
#PBS -l walltime=24:00:00
#PBS -e myprog.err
#PBS -o myprog.out
#PBS -q default-cpu
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/dense_prediction/forked/AdvSemiSeg/
python train_binaryDis_7.py --snapshot-dir snapshots/binaryDis_7_50_low_lr  --restore-from snapshots/binaryDis_7_50_low_lr/VOC_12800.pth --restore-from-D snapshots/binaryDis_7_50_low_lr/VOC_12800_D.pth  --partial-data 0.5  --num-steps 20000  >> out_binaryDis_7_50_low_lr_cont.txt
