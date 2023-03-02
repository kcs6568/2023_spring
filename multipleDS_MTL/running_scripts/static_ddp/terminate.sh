#!/bin/bash
TRAIN_SCRIPT=$1
KILLER="kill $(ps aux | grep "$1" | grep -v grep | awk '{print $2}')"
# KILLER="kill $(ps aux | grep train3.py | grep -v grep | awk '{print $2}')"

$KILLER