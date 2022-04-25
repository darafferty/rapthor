#!/usr/bin/env python3
"""
Script to make an aterm-config file for WSClean
"""
import argparse
from argparse import RawTextHelpFormatter
import sys
from rapthor.lib import miscellaneous as misc


def main(output_file, tec_filenames=None, gain_filenames=None, use_beam=True):
    """
    Make an aterm-config file for WSClean

    Parameters
    ----------
    output_file : str
        Filename of output config file
    tec_filenames : list, optional
        List of filenames for TEC images
    gain_filenames : list, optional
        List of filenames for gain images
    """
    if tec_filenames is None and gain_filenames is None:
        print('make_aterm_config: One of tec_filenames or gain_filenames must be specified')
        sys.exit(1)
    use_beam = misc.string2bool(use_beam)

    terms = []
    if tec_filenames is not None:
        terms.append('tec')
        tec_images = misc.string2list(tec_filenames)
    if gain_filenames is not None:
        terms.append('diagonal')
        gain_images = misc.string2list(gain_filenames)
    if use_beam:
        terms.append('beam')
    aterm_str = 'aterms = [{}]\n'.format(', '.join(terms))
    lines = [aterm_str]
    if tec_filenames is not None:
        lines.append('tec.images = [{}]\n'.format(','.join(tec_images)))
    if gain_filenames is not None:
        lines.append('diagonal.images = [{}]\n'.format(','.join(gain_images)))
    lines.append('beam.differential = true\n')
    lines.append('beam.update_interval = 120\n')
    lines.append('beam.usechannelfreq = true\n')

    config_file = open(output_file, 'w')
    config_file.writelines(lines)
    config_file.close()


if __name__ == '__main__':
    descriptiontext = "Make an a-term configuration file.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('output_file', help='Filename of output config file')
    parser.add_argument('--tec_filenames', help='Filenames of TEC aterm images', type=str, default=None)
    parser.add_argument('--gain_filenames', help='Filenames of gain aterm images', type=str, default=None)
    args = parser.parse_args()
    main(args.output_file, tec_filenames=args.tec_filenames,
         gain_filenames=args.gain_filenames)
