#!/bin/bash -x
docker run -it --rm \
    -v /project/rapthor/Software/rapthor/share/casacore/data:/usr/share/casacore/data:ro \
    -v /project/rapthor/Data/rapthor/N6946:/project/rapthor/Data/rapthor/N6946:ro \
    -v /project/rapthor/Share/rapthor/N6946:/project/rapthor/Share/rapthor/N6946:rw \
    --entrypoint /bin/bash \
    rapthor-debug
