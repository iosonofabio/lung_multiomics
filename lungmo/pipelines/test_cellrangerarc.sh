#!/bin/sh
DATA_PATH=/home/fabio/projects/lung_multiomics/data/pilot_Dec2022/cellranger_output
mkdir -p $DATA_PATH

#cd $DATA_PATH && /home/fabio/software/cellranger-arc-2.0.2/bin/cellranger-arc count --id=nor-1 \
#	             --reference=/home/keyi/refdata-cellranger-arc-mm10-2020-A-2.0.0 \
#		     --libraries=/home/fabio/projects/lung_multiomics/lungmo/pipelines/config/pilot_Dec2022_nor-1.csv \
#		     --localcores=32 \
#		     --localmem=256

cd $DATA_PATH && /home/fabio/software/cellranger-arc-2.0.2/bin/cellranger-arc count --id=nor-2 \
	             --reference=/home/keyi/refdata-cellranger-arc-mm10-2020-A-2.0.0 \
		     --libraries=/home/fabio/projects/lung_multiomics/lungmo/pipelines/config/pilot_Dec2022_nor-2.csv \
		     --localcores=32 \
		     --localmem=256
