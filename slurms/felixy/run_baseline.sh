#!/bin/bash

for input_type in latefusion centerframe randomframe
do
    sbatch run_test.slurm ${input_type} ${input_type}
done
