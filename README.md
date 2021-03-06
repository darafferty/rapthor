Rapthor: LOFAR DDE Pipeline
===========================

Rapthor is an experimental pipeline for correcting direction-dependent effects in LOFAR data. It uses DPPP and WSClean to derive and apply the corrections as smooth 2-D screens. It uses [CWL](https://www.commonwl.org) for the pipeline language and [Toil](http://toil.ucsc-cgl.org) to run the pipelines.


Installation
------------

To install rapthor, follow the instructions below.


### Dependencies

Rapthor requires the following packages (beyond those installed automatically with rapthor):

* [WSClean](http://sourceforge.net/p/wsclean/wiki/Home) (version 2.9 or later; building with [IDG](https://gitlab.com/astron-idg/idg)  and the [LOFAR Beam Library](https://github.com/lofar-astron/LOFARBeam) is required)
* [DP3](https://github.com/lofar-astron/DP3) (version v4.1 or later; building with [Dysco](https://github.com/aroffringa/dysco) and the [LOFAR Beam Library](https://github.com/lofar-astron/LOFARBeam) is required)
* [LSMTool](https://github.com/darafferty/LSMTool) (version 1.4.2 or later)
* [LoSoTo](https://github.com/revoltek/losoto) (version 2.0 or later)
* [PyBDSF](https://github.com/lofar-astron/PyBDSF) (version 1.9.2 or later)

### Downloading and Installing

Get the latest developer version by cloning the git repository:

    git clone https://github.com/darafferty/rapthor.git

Then install with:

    cd rapthor
    python setup.py install


Usage
-----

The `rapthor` executable can be used from the command line with
a parset that defines the parameters of the run. E.g.:

    $ rapthor rapthor.parset

The parset defines the input and working directories, various options, etc. For details,
please see the example parset in the examples directory.
