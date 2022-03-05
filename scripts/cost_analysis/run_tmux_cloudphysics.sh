#!/bin/bash
rdHistPath=/research2/mtc/cp_traces/rd_hist_4k/
workloadIndex=$1
startCost=$2
endCost=$3

for curCost in $(seq $startCost $endCost)
do 
    if [[ $workloadIndex -gt 9 ]]
    then
        echo "launchind tmux for workload w$workloadIndex"
        scriptCommand="python3 /home/pranav/MASCOTS/scripts/cost_analysis/CostAnalysis.py w$workloadIndex $curCost"
        tmux new-session -d -s w$workloadIndex-$curCost $scriptCommand
    else
        echo "launchind tmux for workload w0$workloadIndex"
        scriptCommand="python3 /home/pranav/MASCOTS/scripts/cost_analysis/CostAnalysis.py w0$workloadIndex $curCost"
        tmux new-session -d -s w0$workloadIndex-$curCost $scriptCommand
    fi
done