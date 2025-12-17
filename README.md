# Rapthor: LOFAR DDE Pipeline

Rapthor is an experimental pipeline for correcting direction-dependent effects in LOFAR data. It is also being developed for use on SKA-Low data. It uses DP3 and WSClean to derive and apply the corrections in facets or as smooth 2-D screens. It uses [CWL](https://www.commonwl.org) for the pipeline language and either [Toil](http://toil.ucsc-cgl.org) or [StreamFlow](https://streamflow.di.unito.it) to run the pipelines.

## Documentation

Extensive documentation can be found on [Read the Docs](https://rapthor.readthedocs.io/en/latest/).

For details on how rapthor is installed and used as part of the SKAO ICAL pipeline see SKAO 
documentation [here](https://developer.skao.int/projects/ska-sdp-ical/en/latest/).

## Installation

To install rapthor, follow the instructions below. 

### Dependencies

Rapthor requires the following packages (beyond those installed automatically with rapthor):

* [DP3](https://git.astron.nl/RD/DP3.git) (version 6.5 or later; building with [AOFlagger](https://gitlab.com/aroffringa/aoflagger), [EveryBeam](https://git.astron.nl/RD/EveryBeam), and [IDG](https://git.astron.nl/RD/idg) is required)
* [EveryBeam](https://git.astron.nl/RD/EveryBeam.git) (version 0.7.4 or later)
* [IDG](https://git.astron.nl/RD/idg.git) (version 1.2.0 or later)
* [WSClean](https://gitlab.com/aroffringa/wsclean.git) (version 3.6 or later; building with [EveryBeam](https://git.astron.nl/RD/EveryBeam) and [IDG](https://git.astron.nl/RD/idg) is required)

It is strongly recommended to also build the following packages from source, because the version that can be installed from the Linux distribution is usually too old:

* [AOFlagger](https://gitlab.com/aroffringa/aoflagger.git) (version 3.4.0 or later)
* [Casacore](https://github.com/casacore/casacore.git) (version 3.7.1 or later)
* [Python-Casacore](https://github.com/casacore/python-casacore.git) (version 3.7.1 or later)


### Downloading and Installing

Installation can be done in a number of ways. In order of preference (read:
ease of use):

1. Install the latest release from PyPI:

    ```
    pip install rapthor
    ```

2. Install directly from the Rapthor git repository. This option is useful if you want to use one or more features that have not yet been released:

    ```
    pip install --upgrade pip
    pip install git+https://git.astron.nl/RD/rapthor.git[@<branch|tag|hash>]
    ```
    If the optional `@<branch|tag|hash>` is omitted, `HEAD` of the `master` branch will used.

3. Clone the git repository, and install from your working copy. This option is mostly used by developers who want to make local changes:

    ```
    pip install --upgrade pip
    git clone https://git.astron.nl/RD/rapthor.git
    cd rapthor
    git checkout [<branch|tag|hash>]  #optionally
    pip install .
    ```

#### Note

When installing Rapthor from source (options 2 and 3), you will need `pip` version 23 or later. That is why you need to upgrade to the latest version of `pip` *before* you install from source.


## Usage

The `rapthor` executable can be used from the command line with
a parset that defines the parameters of the run. E.g.:

    $ rapthor rapthor.parset

The parset defines the input and working directories, various options, etc. For details,
please see the example parset in the examples directory.
