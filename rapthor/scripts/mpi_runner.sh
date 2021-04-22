#!/bin/bash -e

unset "${!SLURM@}"

cmd="salloc "
for var in "$@"
do
    cmd+="$var "
done

job=$($cmd)
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Error: problem running salloc command"
    exit $exit_status
fi
