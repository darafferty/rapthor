"""
Initial image operation adapter.
"""

import json
import os

from rapthor.operations.image.base import Image
from rapthor.operations.image.diagnostics import report_sector_diagnostics


class ImageInitial(Image):
    """
    Operation to image the field to generate an initial sky model
    """

    def __init__(self, field):
        super().__init__(field, index=None, name="initial_image")

        self.apply_none = True

    def set_parset_parameters(self):
        """
        Define parameters needed by the initial image flow.
        """
        # Set parameters as needed
        self.apply_screens = False
        self.use_facets = False
        self.save_source_list = True
        self.peel_bright_sources = False
        self.make_image_cube = False
        self.make_residual_visibilities = False
        self.image_pol = "I"
        self.compress_images = self.field.compress_selfcal_images
        super().set_parset_parameters()

    def set_input_parameters(self):
        """
        Define inputs passed to the initial image flow.
        """
        # Set the imaging parameters that are optimal for the initial sky
        # model generation
        self.apply_amplitudes = False
        self.apply_fulljones = False
        self.apply_normalizations = False
        self.field.full_field_sector.auto_mask = 5.0
        self.field.full_field_sector.auto_mask_nmiter = 1
        self.field.full_field_sector.threshisl = 4.0
        self.field.full_field_sector.threshpix = 5.0
        self.field.full_field_sector.max_nmiter = 8
        self.field.full_field_sector.max_wsclean_nchannels = 8
        self.field.full_field_sector.channel_width_hz = 6e6
        self.imaging_sectors = [self.field.full_field_sector]
        self.imaging_parameters = self.field.parset["imaging_specific"].copy()
        self.imaging_parameters["cellsize_arcsec"] = 1.5
        self.imaging_parameters["robust"] = -1.5
        self.imaging_parameters["taper_arcsec"] = 0.0
        self.imaging_parameters["mgain"] = 0.85
        self.imaging_parameters["reweight"] = False
        self.imaging_parameters["dd_psf_grid"] = [1, 1]
        self.do_predict = False
        self.do_multiscale_clean = True
        self.field.disable_clean = False
        self.field.skip_final_major_iteration = True
        super().set_input_parameters()

    def finalize(self):
        """
        Finalize this operation
        """
        sector = self.field.full_field_sector

        # Save the output images
        images = {"sector_I_images", "sector_extra_images"}
        if self.field.save_supplementary_images:
            if self.outputs["source_filtering_mask"][0]:
                images.update({"source_filtering_mask"})
        self.copy_outputs_to(
            os.path.join(self.parset["dir_working"], "images", self.name), include=images, move=True
        )

        # Save the output sky models. We also set the paths as attributes of the sector for later
        # use
        skymodel_dest_dir = os.path.join(self.parset["dir_working"], "skymodels", self.name)
        for skymodel_type in ["true_sky", "apparent_sky"]:
            src_sector_skymodel = self.outputs[f"filtered_skymodel_{skymodel_type}"][0]["path"]
            sector_skymodel_file = os.path.join(
                skymodel_dest_dir, os.path.basename(src_sector_skymodel)
            )
            setattr(sector, f"image_skymodel_file_{skymodel_type}", sector_skymodel_file)
            self.copy_outputs_to(
                skymodel_dest_dir,
                include={f"filtered_skymodel_{skymodel_type}"},
                move=True,
            )

        # Save the output PyBDSF source catalog
        self.copy_outputs_to(skymodel_dest_dir, include={"pybdsf_catalog"}, move=True)

        # Save the astrometry and photometry plots and diagnostics file
        diagnostics_dest_dir = os.path.join(
            os.path.join(self.parset["dir_working"], "plots", self.name)
        )
        diagnotics = {"sector_diagnostics"}
        if self.outputs["sector_diagnostic_plots"][0]:
            diagnotics.update({"sector_diagnostic_plots"})
        self.copy_outputs_to(diagnostics_dest_dir, include=diagnotics, move=True)

        # Read in the image diagnostics and log a summary of them
        diagnostics_file = os.path.join(
            diagnostics_dest_dir,
            os.path.basename(self.outputs["sector_diagnostics"][0]["path"]),
        )
        with open(diagnostics_file, "r") as f:
            diagnostics_dict = json.load(f)
        sector.diagnostics.append(diagnostics_dict)
        ratio, std = report_sector_diagnostics(sector.name, diagnostics_dict, self.log)
        self.field.lofar_to_true_flux_ratio = ratio
        self.field.lofar_to_true_flux_std = std

        # Clean up other files
        self.clean_outputs()

        # Finally call finalize() of the Operation class
        super(Image, self).finalize()
