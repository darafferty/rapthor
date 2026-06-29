#!/usr/bin/env python3
"""CLI wrapper for calculating flux-scale normalization corrections."""

from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.execution.image.flux_normalization import normalize_flux_scale


def main(
    source_catalog,
    ms_file,
    output_h5parm,
    radius_cut=3.0,
    major_axis_cut=30 / 3600,
    neighbor_cut=30 / 3600,
    spurious_match_cut=30 / 3600,
    min_sources=5,
    weight_by_flux_err=False,
    ignore_frequency_dependence=False,
    reference_skymodels=None,
    reference_skymodels_frequencies=None,
):
    """Calculate flux-scale normalization corrections."""
    normalize_flux_scale(
        source_catalog,
        ms_file,
        output_h5parm,
        radius_cut=radius_cut,
        major_axis_cut=major_axis_cut,
        neighbor_cut=neighbor_cut,
        spurious_match_cut=spurious_match_cut,
        min_sources=min_sources,
        weight_by_flux_err=weight_by_flux_err,
        ignore_frequency_dependence=ignore_frequency_dependence,
        reference_skymodels=reference_skymodels,
        reference_skymodels_frequencies=reference_skymodels_frequencies,
    )


if __name__ == "__main__":
    descriptiontext = "Calculate flux-scale normalization corrections.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("source_catalog", help="Filename of input FITS source catalog")
    parser.add_argument("ms_file", help="Filename of imaging MS file")
    parser.add_argument(
        "output_h5parm", help="Filename of output H5parm file with the normalization corrections"
    )
    parser.add_argument("--radius_cut", help="Radius cut in degrees", type=float, default=3.0)
    parser.add_argument(
        "--major_axis_cut", help="Major-axis size cut in degrees", type=float, default=30 / 3600
    )
    parser.add_argument(
        "--neighbor_cut",
        help="Nearest-neighbor distance cut in degrees",
        type=float,
        default=30 / 3600,
    )
    parser.add_argument(
        "--spurious_match_cut",
        help="Spurious match distance cut in degrees",
        type=float,
        default=30 / 3600,
    )
    parser.add_argument(
        "--min_sources",
        help="Minimum number of sources required for normalization calculation",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--weight_by_flux_err",
        help="Weight by error on flux density",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ignore_frequency_dependence",
        help="Ignore frequency dependence of normalizations",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--reference_skymodels",
        help=(
            "Filenames of reference sky models to use for normalization "
            "(instead of external survey catalogs)"
        ),
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--reference_skymodels_frequencies",
        help="Frequencies corresponding to the reference sky models",
        type=float,
        nargs="+",
        default=None,
    )

    args = parser.parse_args()
    main(
        args.source_catalog,
        args.ms_file,
        args.output_h5parm,
        radius_cut=args.radius_cut,
        major_axis_cut=args.major_axis_cut,
        neighbor_cut=args.neighbor_cut,
        spurious_match_cut=args.spurious_match_cut,
        min_sources=args.min_sources,
        weight_by_flux_err=args.weight_by_flux_err,
        ignore_frequency_dependence=args.ignore_frequency_dependence,
        reference_skymodels=args.reference_skymodels,
        reference_skymodels_frequencies=args.reference_skymodels_frequencies,
    )
