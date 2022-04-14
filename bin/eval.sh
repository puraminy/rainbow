alias ft="python /home/pouramini/rainbow/bin/fine-tune.py"
if [ -z $1 ]; then
   echo "Provide method and test numbers and relation"
else
   if [ -z $3 ]; then
	rel=all
   else
	rel=$3
   fi
   for d in 10*/ ; do
      echo "${PWD}/$d"
      if [ -z $4 ]; then
	      ft --do_eval --do_score -mt $1 -md "${PWD}/$d" -tn $2 --rel_filter=$rel
      else
	      ft --do_score -mt $1 -md "${PWD}/$d" -tn $2 --rel_filter=$rel
      fi
   done
fi

