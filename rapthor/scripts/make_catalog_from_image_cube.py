#!/usr/bin/env python3
"""
Script to make a source catalog from an image cube
"""

from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.execution.image.cubes import make_catalog_from_image_cube


def main(
    cube_image,
    cube_beams,
    cube_frequencies,
    output_catalog,
    threshisl=3.0,
    threshpix=5.0,
    rmsbox=(150, 50),
    rmsbox_bright=(35, 7),
    adaptive_thresh=75.0,
    ncores=8,
):
    """
    Make a source catalog from an image cube

    Parameters
    ----------
    cube_image : str
        Filename of input FITS cube image to use to detect sources
    cube_beams : str
        Filename of input text file with cube beam parameters. The file should
        give the beams as written as "(major axis, minor axis, position angle)"
        in degrees, one per cube channel. The beams for all channels should be
        given on a single line, separated by commas. E.g.:
            (0.0091, 0.0073, 38.1526), (0.0090, 0.0074, 39.1030), ...
    cube_frequencies : str
        Filename of input text file with cube frequency parameters. The file
        should give the frequencies in Hz, one per cube channel. The frequencies
        for all channels should be given on a single line, separated by commas.
        E,g.:
            23143005.3710, 129002380.3710, ...
    output_catalog : str
        Filename of output FITS source catalog
    threshisl : float, optional
        Value of thresh_isl PyBDSF parameter
    threshpix : float, optional
        Value of thresh_pix PyBDSF parameter
    rmsbox : tuple of floats, optional
        Value of rms_box PyBDSF parameter
    rmsbox_bright : tuple of floats, optional
        Value of rms_box_bright PyBDSF parameter
    adaptive_thresh : float, optional
        This value sets the threshold above which a source will use the small
        rms box
    ncores : int, optional
        Maximum number of cores to use
    """
    make_catalog_from_image_cube(
        cube_image,
        cube_beams,
        cube_frequencies,
        output_catalog,
        threshisl=threshisl,
        threshpix=threshpix,
        rmsbox=rmsbox,
        rmsbox_bright=rmsbox_bright,
        adaptive_thresh=adaptive_thresh,
        ncores=ncores,
    )


if __name__ == "__main__":
    descriptiontext = "Make a source catalog from an image cube.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("cube_image", help="Filename of input FITS image cube")
    parser.add_argument("cube_beams", help="Filename of input text file with cube beam parameters")
    parser.add_argument(
        "cube_frequencies",
        help="Filename of input text file with cube frequency parameters",
    )
    parser.add_argument("output_catalog", help="Filename of output FITS catalog")
    parser.add_argument("--threshisl", help="Island threshold", type=float, default=3.0)
    parser.add_argument("--threshpix", help="Peak pixel threshold", type=float, default=5.0)
    parser.add_argument(
        "--rmsbox",
        help='Rms box width and step (e.g., "(60, 20)")',
        type=str,
        default="(150, 50)",
    )
    parser.add_argument(
        "--rmsbox_bright",
        help='Rms box for bright sources, width and step (e.g., "(60, 20)")',
        type=str,
        default="(35, 7)",
    )
    parser.add_argument("--adaptive_thresh", help="Adaptive threshold", type=float, default=75.0)
    parser.add_argument("--ncores", help="Max number of cores to use", type=int, default=8)

    args = parser.parse_args()
    main(
        args.cube_image,
        args.cube_beams,
        args.cube_frequencies,
        args.output_catalog,
        threshisl=args.threshisl,
        threshpix=args.threshpix,
        rmsbox=args.rmsbox,
        rmsbox_bright=args.rmsbox_bright,
        adaptive_thresh=args.adaptive_thresh,
        ncores=args.ncores,
    )
