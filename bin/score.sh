alias ft="python /home/pouramini/rainbow/bin/fine-tune.py"
if [ -z $1 ]; then
   echo "Please provide method"
else
   for d in 10*/ ; do
       echo "${PWD}/$d"
       ft --do_score -mt $1 -md "${PWD}/$d" -tn $2
   done
fi

