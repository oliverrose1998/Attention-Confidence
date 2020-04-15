#!/bin/tcsh

set SCRIPT=run_script.sh
set year=`date +'%Y'`
set month=`date +'%m'`
set day=`date +'%d'`
set log_file="LOG/${year}_${month}_${day}_out.txt"

echo  "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >>  $log_file
echo  "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >>  $log_file
echo  `date` >> $log_file

qsub -P babel -l ubuntu=1 -l qp=low -l osrel="*" -o $log_file -j y $SCRIPT
