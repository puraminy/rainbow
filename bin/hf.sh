#!/bin/bash
. /home/pouramini/aa
alias py=python
if [ -z $1 ]; then
   echo "Provide model type"
else
   folder=${PWD##*/}          # to assign to a variable
   if [ -z $2 ]; then
	step=900          # to assign to a variable
   else
	step=$2
   fi
   if [ $1 = "t5-base" ]; then
	   echo $1
	   hfconv -ckp 999900 series -s $step -b t5-base
	   caldif hf/hf* -fid t5-base
	   heatmap htsv -f $dest/$folder
   elif [ $1 = "t5-large" ]; then
	   hfconv -ckp 1000700 series -s $step -b t5-large
	   caldif hf/hf* -fid t5-large
	   heatmap htsv -f $dest/$folder
   fi
fi
