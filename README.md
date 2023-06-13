Rapthor: LOFAR DDE Pipeline
===========================

Rapthor is an experimental pipeline for correcting direction-dependent effects in LOFAR data. It uses DPPP and WSClean to derive and apply the corrections in facets or as smooth 2-D screens. It uses [CWL](https://www.commonwl.org) for the pipeline language and [Toil](http://toil.ucsc-cgl.org) to run the pipelines.

## Documentation

Extensive documentation can be found on [Read the Docs](https://rapthor.readthedocs.io/en/latest/).


Installation
------------

To install rapthor, follow the instructions below.


### Dependencies

Rapthor requires the following packages (beyond those installed automatically with rapthor):

* [WSClean](https://gitlab.com/aroffringa/wsclean) (version 2.9 or later; building with [IDG](https://gitlab.com/astron-idg/idg) and [EveryBeam](https://git.astron.nl/RD/EveryBeam) is required)
* [DP3](https://git.astron.nl/RD/DP3) (version v4.1 or later; building with [Dysco](https://github.com/aroffringa/dysco) and [EveryBeam](https://git.astron.nl/RD/EveryBeam) is required)
* [LSMTool](https://git.astron.nl/RD/LSMTool) (version 1.4.2 or later)
* [LoSoTo](https://github.com/revoltek/losoto) (version 2.0 or later)
* [PyBDSF](https://github.com/lofar-astron/PyBDSF) (version 1.9.2 or later)

### Downloading and Installing

Installation can be done in a number of ways. In order of preference (read:
easy of use):

- Install the latest release from PyPI:

  ```
  pip install rapthor
  ```

- Install directly from the Rapthor git repository. This option is useful if you want to use one or more features that have not yet been released:

  ```
  pip install git+https://git.astron.nl/RD/rapthor.git[@<branch|tag|hash>]
  ```

  If the optional `@<branch|tag|hash>` is omitted, `HEAD` of the `master` branch will used.

- Clone the git repository, and install from your working copy. This option is mostly used by developers who want to make local changes:

  ```
  git clone https://git.astron.nl/RD/rapthor.git
  cd rapthor
  git checkout [<branch|tag|hash>]  #optionally
  pip install .
  ```

Usage
-----

The `rapthor` executable can be used from the command line with
a parset that defines the parameters of the run. E.g.:

    $ rapthor rapthor.parset

The parset defines the input and working directories, various options, etc. For details,
please see the example parset in the examples directory.
