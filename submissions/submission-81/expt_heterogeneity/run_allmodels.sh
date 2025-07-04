#!/bin/bash

for task_id in {1..28}
do
  echo "Running task_id=$task_id"
  python train_saes.py --task_id $task_id
done
