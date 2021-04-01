#!/bin/bash -e

while getopts ":m:n:k:c:z:i:j:r:u:v:x:s:l:o:d:t:p:a:g:y:q:b:h:" arg; do
  case $arg in
    m) msin=$OPTARG;;
    n) name=$OPTARG;;
    k) mask=$OPTARG;;
    c) config=$OPTARG;;
    z) imsize=$OPTARG;;
    i) niter=$OPTARG;;
    j) nmiter=$OPTARG;;
    r) robust=$OPTARG;;
    u) min_uv_lambda=$OPTARG;;
    v) max_uv_lambda=$OPTARG;;
    x) cellsize_deg=$OPTARG;;
    s) multiscale_scales_pixel=$OPTARG;;
    l) dir_local=$OPTARG;;
    o) channels_out=$OPTARG;;
    d) deconvolution_channels=$OPTARG;;
    t) taper_arcsec=$OPTARG;;
    p) mem=$OPTARG;;
    a) auto_mask=$OPTARG;;
    g) idg_mode=$OPTARG;;
    y) ncpus_per_task=$OPTARG;;
    q) nnodes=$OPTARG;;
    b) numthreads=$OPTARG;;
    h) numdecthreads=$OPTARG;;
  esac
done

# build the mpirun command
infix=$(cat /dev/urandom | env LC_ALL=C tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
mpi_command="mpirun -np \$SLURM_JOB_NUM_NODES --pernode --prefix \$MPI_PREFIX -x LD_LIBRARY_PATH=\$LD_LIBRARY_PATH wsclean-mp -no-update-model-required -multiscale -save-source-list -local-rms -join-channels -use-idg -mem ${mem} -j ${numthreads} -deconvolution-threads ${numdecthreads} -pol I -mgain 0.85 -log-time -fit-spectral-pol 3 -multiscale-shape gaussian -auto-threshold 1.0 -local-rms-window 50 -local-rms-method rms-with-min -aterm-kernel-size 32 -weight briggs ${robust} -name ${name} -fits-mask ${mask} -aterm-config ${config} -size ${imsize} -niter ${niter} -nmiter ${nmiter} -minuv-l ${min_uv_lambda} -maxuv-l ${max_uv_lambda} -scale ${cellsize_deg} -multiscale-scales ${multiscale_scales_pixel} -parallel-deconvolution 2048 -temp-dir ${dir_local} -channels-out ${channels_out} -deconvolution-channels ${deconvolution_channels} -taper-gaussian ${taper_arcsec} -auto-mask ${auto_mask} -idg-mode ${idg_mode} ${msin}"

# make sbatch file
exec 3<> wsclean_mpi_$infix.slurm
    echo "#!/bin/bash -e" >&3
    echo "#SBATCH --job-name=mpijob" >&3
    echo "#SBATCH --time=4-00:00:00" >&3
    echo "#SBATCH --nodes=${nnodes}" >&3
    echo "#SBATCH --ntasks-per-node=1" >&3
    echo "#SBATCH --cpus-per-task=${ncpus_per_task}" >&3
    echo "#SBATCH --output output_${infix}.log" >&3
    echo "cd \$SLURM_SUBMIT_DIR" >&3
    echo $mpi_command >&3
exec 3>&-

# unset SLURM environment variables
unset "${!SLURM@}"

# run sbatch
job=$(sbatch wsclean_mpi_$infix.slurm)
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Error: problem running sbatch command"
    exit $exit_status
fi

# loop while batch job is in the queue; wait for it to finish
jobid=$(echo $job | grep -o '[[:digit:]]\+$')
while squeue -ho%i -j $jobid 2>/dev/null | grep -q "^$jobid$"
do
    sleep 1
done

# Fetch ExitCode of batch job "n:m"; where "n" is exit code of the script,
# and "m" is the signal that terminated the process (if it was signalled).
# Return 128+m as status if process was signalled, otherwise return n.
exitcode=$(sacct -no ExitCode -j ${jobid}.batch)
if [ -z "$exitcode" ]; then
    echo "Error: failed to fetch accounting information for job $jobid"
    exit 1
else
    status=${exitcode%:*}
    signal=${exitcode#*:}
    if [ $signal -ne 0 ]; then
        exit $((128 + $signal))
    else
        exit $status
    fi
fi
