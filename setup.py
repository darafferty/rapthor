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
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 1 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    install_requires=['numpy', 'scipy', 'astropy', 'jinja2', 'shapely', 'toil[cwl]', 'Pillow', 'reproject', 'python-dateutil', 'pytz'],
    scripts=['bin/rapthor', 'bin/plotrapthor'],
    packages=['rapthor', 'rapthor.operations', 'rapthor.lib'],
    package_data={'rapthor': [
        'pipeline/parsets/*',
        'pipeline/steps/*',
        'scripts/*',
        'skymodels/*']},
    cmdclass={'test': PyTest},
    )
