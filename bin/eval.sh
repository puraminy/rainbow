alias ft="python /home/pouramini/rainbow/bin/fine-tune.py"
if [ -z $1 ]; then
   echo "Provide method and test numbers"
else
   for d in 10*/ ; do
      echo "${PWD}/$d"
      cp operative_config.gin $d
      ft --do_eval -mt $1 -md "${PWD}/$d" -tn $2
   done
fi

