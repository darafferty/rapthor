"""
Flux-scale normalization image operation adapter.
"""

import os

from rapthor.lib.records import FileRecord
from rapthor.operations.image.base import Image


class ImageNormalize(Image):
    """
    Operation to image for flux-scale normalization
    """

    def __init__(self, field, index):
        super().__init__(field, index=index, name="normalize")

        self.normalization_skymodels = None
        self.normalization_reference_frequencies = None

    def set_parset_parameters(self):
        """
        Define parameters needed by the normalization image flow.
        """
        # Set parameters as needed
        self.save_source_list = False
        self.peel_bright_sources = False
        self.make_image_cube = True
        self.normalize_flux_scale = True
        self.compress_images = self.field.compress_selfcal_images
        self.image_cube_stokes_list = ["I"]
        if self.field.h5parm_filename is None:
            # No calibration has yet been done, so set various flags as needed
            self.use_facets = False
            self.apply_screens = False
        if self.normalization_skymodels is None:
            self.normalization_skymodels = self.field.normalization_skymodels
            self.normalization_reference_frequencies = (
                self.field.normalization_reference_frequencies
            )
        super().set_parset_parameters()
        self.parset_parms.update(
            {
                "normalization_skymodels": self.normalization_skymodels,
                "normalization_reference_frequencies": self.normalization_reference_frequencies,
            }
        )

    def set_input_parameters(self):
        """
        Define inputs passed to the normalization image flow.
        """
        # Set the imaging parameters that are optimal for the flux-scale
        # normalization
        self.apply_none = self.field.h5parm_filename is None
        self.apply_normalizations = False
        self.field.normalize_sector.auto_mask = 5.0
        self.field.normalize_sector.auto_mask_nmiter = 2
        self.field.normalize_sector.threshisl = 4.0
        self.field.normalize_sector.threshpix = 5.0
        self.field.normalize_sector.max_nmiter = 8
        self.field.normalize_sector.max_wsclean_nchannels = 8
        self.field.normalize_sector.channel_width_hz = 4e6
        self.imaging_sectors = [self.field.normalize_sector]
        self.imaging_parameters = self.field.parset["imaging_specific"].copy()
        self.imaging_parameters["cellsize_arcsec"] = 6.0
        self.imaging_parameters["robust"] = -0.5
        self.imaging_parameters["taper_arcsec"] = 24.0
        self.do_predict = False
        self.do_multiscale_clean = False
        self.field.disable_clean = False
        self.field.skip_final_major_iteration = False
        super().set_input_parameters()
        self.input_parms.update(
            {
                "normalization_skymodels": [
                    FileRecord(filename).to_json()
                    for filename in self.normalization_skymodels or ()
                ]
                or None,
                "normalization_reference_frequencies": self.normalization_reference_frequencies,
            }
        )

    def finalize(self):
        """
        Finalize this operation
        """
        # Save the output h5parm with the flux-scale corrections
        src_filename = self.outputs["sector_normalize_h5parm"][0]["path"]
        dest_dir = os.path.join(self.parset["dir_working"], "solutions", self.name)
        self.field.normalize_h5parm = os.path.join(dest_dir, os.path.basename(src_filename))
        self.copy_outputs_to(dest_dir, include={"sector_normalize_h5parm"}, move=True)

        # Save the output image cubes
        image_cube_keys = {
            "sector_image_cubes",
            "sector_image_cube_beams",
            "sector_image_cube_frequencies",
        }
        dest_dir = os.path.join(self.parset["dir_working"], "images", self.name)
        self.copy_outputs_to(dest_dir, include=image_cube_keys, move=True)

        # Clean up other files
        self.clean_outputs()

        # Set the flags for subsequent processing
        self.field.normalize_flux_scale = False
        self.field.apply_normalizations = True

        # Finally call finalize() of the Operation class
        super(Image, self).finalize()
