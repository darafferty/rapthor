#!/bin/bash

while getopts ":m:n:k:c:z:i:j:r:p:u:v:x:s:l:o:d:t:a:g:y:q:h:" arg; do
  case $arg in
    m) msin=$OPTARG;;
    n) name=$OPTARG;;
    k) mask=$OPTARG;;
    c) config=$OPTARG;;
    z) imsize=$OPTARG;;
    i) niter=$OPTARG;;
    j) nmiter=$OPTARG;;
    r) robust=$OPTARG;;
    p) padding=$OPTARG;;
    u) min_uv_lambda=$OPTARG;;
    v) max_uv_lambda=$OPTARG;;
    x) cellsize_deg=$OPTARG;;
    s) multiscale_scales_pixel=$OPTARG;;
    l) dir_local=$OPTARG;;
    o) channels_out=$OPTARG;;
    d) deconvolution_channels=$OPTARG;;
    t) taper_arcsec=$OPTARG;;
    a) auto_mask=$OPTARG;;
    g) idg_mode=$OPTARG;;
    y) ntasks=$OPTARG;;
    q) nnodes=$OPTARG;;
    h) hosts=$OPTARG;;
  esac
done

# make the hostfile from the supplied list of hosts
makehostfile hostfile_${infix}.txt --hosts=${hosts}

# run WSClean
infix=$(cat /dev/urandom | env LC_ALL=C tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
mpirun -np ${nnodes} -hostfile hostfile_${infix}.txt --pernode --prefix \$MPI_PREFIX -x LD_LIBRARY_PATH=\$LD_LIBRARY_PATH wsclean-mp -no-update-model-required -multiscale -fit-beam -reorder -save-source-list -local-rms -join-channels -use-idg -pol I -mgain 0.6 -deconvolution-channels 4 -fit-spectral-pol 3 -multiscale-shape gaussian -weighting-rank-filter 3 -auto-threshold 1.0 -local-rms-window 50 -local-rms-method rms-with-min -aterm-kernel-size 32 -weight briggs ${robust} -name ${name} -fits-mask ${mask} -aterm-config ${config} -size ${imsize} -niter ${niter} -nmiter ${nmiter} -padding ${padding} -minuv-l ${min_uv_lambda} -maxuv-l ${max_uv_lambda} -scale ${cellsize_deg} -multiscale-scales ${multiscale_scales_pixel} -temp-dir ${dir_local} -channels-out ${channels_out} -deconvolution-channels ${deconvolution_channels} -taper-gaussian ${taper_arcsec} -auto-mask ${auto_mask} -idg-mode ${idg_mode} ${msin}
