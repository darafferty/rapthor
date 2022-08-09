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
    url='http://github.com/darafferty/rapthor/',
    description=description,
    long_description=long_description,
    platforms='any',
    classifiers=['Programming Language :: Python :: 3',
                 'Development Status :: 1 - Alpha',
                 'Natural Language :: English',
                 'Intended Audience :: Science/Research',
                 'Operating System :: POSIX :: Linux',
                 'Topic :: Scientific/Engineering :: Astronomy'],
    install_requires=['numpy', 'scipy', 'astropy', 'jinja2', 'shapely',
                      'toil[cwl]!=5.6', 'reproject', 'python-dateutil',
                      'Rtree', 'loess', 'lsmtool', 'losoto', 'bdsf',
                      'python-casacore'],
    scripts=['bin/rapthor',
             'bin/plotrapthor',
             'bin/concat_linc_files',
             'rapthor/scripts/blank_image.py',
             'rapthor/scripts/combine_h5parms.py',
             'rapthor/scripts/filter_skymodel.py',
             'rapthor/scripts/make_aterm_images.py',
             'rapthor/scripts/make_mosaic.py',
             'rapthor/scripts/make_mosaic_template.py',
             'rapthor/scripts/process_slow_gains.py',
             'rapthor/scripts/regrid_image.py',
             'rapthor/scripts/subtract_sector_models.py',
             'rapthor/scripts/split_h5parms.py',
             'rapthor/scripts/mpi_runner.sh'],
    packages=['rapthor', 'rapthor.operations', 'rapthor.lib'],
    package_data={'rapthor': ['pipeline/parsets/*',
                              'pipeline/steps/*',
                              'scripts/*',
                              'skymodels/*']},
    cmdclass={'test': PyTest})
