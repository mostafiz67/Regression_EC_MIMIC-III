#!/bin/bash
if [ -z "$1" ]
then
    echo "Currently listening ports:"
    ss -ltp | grep ssh | grep 127.0.0.1
    echo "Run \`$0 --kill\` to kill all"
else
    ss -ltp | grep ssh | grep 127.0.0.1 | awk '{print $6}' | cut -f 2 -d "," | cut -f 2 -d "=" | xargs kill
fi

