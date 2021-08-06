#!/bin/bash

# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Author: Jakob Maljaars
# Email: jakob.maljaars_@_stcorp.nl

# Script for downloading a mock LOFAR Measurement set

set -e

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_PATH

# Move up to parent folder containing the source
cd ..
mkdir -p test/integration/tmp/data
cd test/integration/tmp/data

MOCK_MODELIMAGE=modelimage.fits

if [ ! -f "$MOCK_MODELIMAGE" ]; then
    wget -q https://www.astron.nl/citt/ci_data/IDG/modelimage.fits -O $MOCK_MODELIMAGE
fi
