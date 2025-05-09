#!/bin/bash
if [ -z "$1" ]
then
    echo "usage: ./watch_cc_jobs NODELIST [LOCAL PORT = 6006] [REMOTE PORT = 6006] ('NODELIST' available from \`sq\` on Compute Canada)"
    exit
else
    NODELIST=$1
    if [ -z "$2" ]
    then
        LOCAL=6006
    else
        LOCAL=$2
    fi
    if [ -z "$3" ]
    then
        REMOTE=6006
    else
        REMOTE=$3
    fi
fi

echo "Connecting to Compute Canada Graham node $NODELIST at port $REMOTE..."
echo "Tensorboard will be available shortly at http://localhost:$LOCAL"
echo "http://localhost:$LOCAL/"
ssh -N -L "localhost:$LOCAL:$NODELIST:$REMOTE" graham
