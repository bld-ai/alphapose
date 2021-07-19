#!/bin/bash
max_retry=5

while true;
do
    python setup.py build develop
    if [[ $? -eq 0 || $max_retry -eq 0 ]]; then
        if [[ $max_retry -eq 0 ]]; then
            echo "max retries reached"
        else
            echo "successfully finished setup with $max_retry attempts left"
        fi
        break
    fi
    ((--max_retry))
done