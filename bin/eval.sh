alias ft="python /home/pouramini/rainbow/bin/fine-tune.py"
if [ -z $1 ]; then
   echo "usage: eval method test_numbers relation pm [+ do_score/ - do_eval and score]"
else
   if [ -z $3 ]; then
	rel=all
   else
	rel=$3
   fi
   for d in tf*/ ; do
      echo "Evaluating ... ${PWD}/$d"
      if [ -z $5 ]; then
	      ft --do_eval --do_score -mt $1 -md "${PWD}/$d" -tn $2 --rel_filter=$rel -pm $4
      else
	      ft --do_score -mt $1 -md "${PWD}/$d" -tn $2 --rel_filter=$rel -pm $4
      fi
   done
fi

