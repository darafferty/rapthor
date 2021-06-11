#!/bin/bash -e

unset "${!SLURM@}"

cmd="salloc "
for var in "$@"
do
    cmd+="$var "
done

eval $cmd
