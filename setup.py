from setuptools import setup, Command
import os
import rapthor._version

description = 'Rapthor: LOFAR DDE Pipeline'
long_description = description
if os.path.exists('README.md'):
    with open('README.md') as f:
        long_description = f.read()


class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys
        import subprocess
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


setup(
    name='rapthor',
    version=rapthor._version.__version__,
    url='https://git.astron.nl/RD/rapthor',
    description=description,
    long_description=long_description,
    platforms='any',
    classifiers=['Programming Language :: Python :: 3',
                 'Development Status :: 1 - Alpha',
                 'Natural Language :: English',
                 'Intended Audience :: Science/Research',
                 'Operating System :: POSIX :: Linux',
                 'Topic :: Scientific/Engineering :: Astronomy'],
    # toil[cwl]!=5.6 because of https://github.com/DataBiosphere/toil/issues/4101
    # toil[cwl]<5.8 because of https://gitter.im/bd2k-genomics-toil/Lobby?at=63d7f912ec2bfc62867d36e6
    # toil[cwl]<5.9 because of https://support.astron.nl/jira/browse/RAP-309
    install_requires=['numpy', 'scipy', 'astropy', 'jinja2', 'shapely',
                      'toil[cwl]!=5.6,<5.8', 'reproject', 'python-dateutil',
                      'Rtree', 'lsmtool', 'losoto', 'bdsf',
                      'python-casacore'],
    scripts=['bin/rapthor',
             'bin/plotrapthor',
             'bin/concat_linc_files',
             'rapthor/scripts/adjust_h5parm_sources.py',
             'rapthor/scripts/blank_image.py',
             'rapthor/scripts/combine_h5parms.py',
             'rapthor/scripts/concat_ms.py',
             'rapthor/scripts/filter_skymodel.py',
             'rapthor/scripts/make_aterm_images.py',
             'rapthor/scripts/make_region_file.py',
             'rapthor/scripts/make_mosaic.py',
             'rapthor/scripts/make_mosaic_template.py',
             'rapthor/scripts/process_slow_gains.py',
             'rapthor/scripts/regrid_image.py',
             'rapthor/scripts/subtract_sector_models.py',
             'rapthor/scripts/split_h5parms.py',
             'rapthor/scripts/mpi_runner.sh'],
    packages=['rapthor', 'rapthor.operations', 'rapthor.lib'],
    package_data={'rapthor': ['pipeline/*/*.cwl',
                              'scripts/*',
                              'skymodels/*']},
    cmdclass={'test': PyTest})
