#!/bin/bash -x
mkdir -p /tmp/loose/rapthor-debug
docker run -it --rm \
    -v /project/rapthor/Share/rapthor/run_480ch_5chunks_flag:/project/rapthor/Share/rapthor/run_480ch_5chunks_flag:ro \
    -v /project/rapthor/Share/rapthor/L632477_480ch_50min:/project/rapthor/Share/rapthor/L632477_480ch_50min:ro \
    -v /tmp/loose/rapthor-debug:/tmp:rw \
    --entrypoint /bin/bash \
    rapthor-debug
