#!/bin/bash
. /home/pouramini/aa
alias py=python
if [ -z $3 ]; then
   echo "Provide model type b) t5_base l)t5_large"
   mtype=l
else
   mtype=$3
fi
if [ -z $1 ]; then
   echo "Provide exp step model_type"
else
   folder=${PWD##*/}          # to assign to a variable
   if [ -z $2 ]; then
	   echo "Provide step"
   else
	   step=$2
	   echo $1
	   if [ $mtype = "b" ]; then
		   hfconv -ckp 999900 series -s $step -b t5-base -hn $1
		   caldif $1/hf* -fid t5-base
		   heatmap htsv -f $dest/$folder --path="${PWD}"/$1
	   elif [ $mtype = "l" ]; then
		   hfconv -ckp 1000700 series -s $step -b t5-large -hn $1
		   caldif $1/hf* -fid t5-large
		   heatmap htsv -f $dest/$folder --path="${PWD}"/$1
	   fi
   fi
fi
