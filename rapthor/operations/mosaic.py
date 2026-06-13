"""
Module that holds the Mosaic class
"""

import os
import shutil

from rapthor.execution.flows.mosaic import mosaic_flow, mosaic_payload_from_inputs
from rapthor.lib.operation import Operation
from rapthor.lib.records import FileRecord


class Mosaic(Operation):
    """
    Operation to mosaic sector images
    """

    def __init__(self, field, index):
        super().__init__(field, index=index, name="mosaic")

        # Determine whether processing is needed
        self.skip_processing = len(self.field.imaging_sectors) < 2

    def set_parset_parameters(self):
        """
        Define parameters needed by the mosaic flow.
        """
        self.parset_parms = self.flow_parset_parameters(
            include_pipeline_working_dir=True,
            skip_processing=self.skip_processing,
            compress_images=self.field.compress_images,
        )

    def set_input_parameters(self):
        """
        Define inputs passed to the mosaic flow.
        """
        # Define various input and output filenames
        sector_image_filename = []
        sector_vertices_filename = []
        regridded_image_filename = []
        template_image_filename = []
        self.image_names = []  # list of input image names
        for pol in self.field.image_pol:
            polup = pol.upper()
            self.image_names.extend(
                [f"{polup}_image_file_true_sky", f"{polup}_image_file_apparent_sky"]
            )
            if not self.field.disable_clean:
                self.image_names.extend(
                    [
                        f"{polup}_model_file_true_sky",
                        f"{polup}_residual_file_apparent_sky",
                        f"{polup}_dirty_file_apparent_sky",
                    ]
                )
        if self.field.save_supplementary_images:
            self.image_names.append("filtering_mask_file")
        if self.field.parset["imaging_specific"]["save_filtered_model_image"]:
            self.image_names.append("filtered_model_file_apparent_sky")

        for image_name in self.image_names:
            image_list = []
            vertices_list = []
            regridded_list = []
            for sector in self.field.imaging_sectors:
                image_list.append(getattr(sector, image_name))
                vertices_list.append(sector.vertices_file)
                regridded_list.append(f"{os.path.basename(getattr(sector, image_name))}.regridded")
            sector_image_filename.append(FileRecord(image_list).to_json())
            sector_vertices_filename.append(FileRecord(vertices_list).to_json())
            regridded_image_filename.append(regridded_list)
            template_image_filename.append(f"{self.name}_template.fits")

        self.mosaic_filename = []
        if self.skip_processing:
            if len(self.field.imaging_sectors) > 0:
                # Use unprocessed files as mosaic files
                for image_name in self.image_names:
                    self.mosaic_filename.append(getattr(self.field.imaging_sectors[0], image_name))
            else:
                self.mosaic_filename.append(None)
        else:
            for image_name in self.image_names:
                # Define output filenames for each mosaic image
                suffix = getattr(self.field.imaging_sectors[0], image_name).split("sector_1")[-1]
                if suffix.endswith(".fz"):
                    # Remove the compressed extension, as the output mosaic files are not
                    # compressed until a later step in the pipeline
                    suffix = os.path.splitext(suffix)[0]
                self.mosaic_filename.append(f"{self.name}{suffix}")

        self.input_parms = {
            "skip_processing": self.skip_processing,
            "sector_image_filename": sector_image_filename,
            "sector_vertices_filename": sector_vertices_filename,
            "template_image_filename": template_image_filename,
            "regridded_image_filename": regridded_image_filename,
            "mosaic_filename": self.mosaic_filename,
        }

    def execute_workflow(self):
        """
        Execute mosaicking through the Prefect flow and return operation outputs.
        """
        payload = mosaic_payload_from_inputs(
            self.input_parms,
            self.pipeline_working_dir,
            compress_images=self.field.compress_images,
        )
        outputs = self.run_prefect_flow(mosaic_flow, payload)
        return True, outputs

    def finalize(self):
        """
        Finalize this operation
        """
        for i, image_name in enumerate(self.image_names):
            if self.mosaic_filename[i] is None:
                # No imaging sectors
                continue
            if not self.skip_processing and self.field.compress_images:
                # Add ".fz" to the filename, since the mosaic image was compressed
                self.mosaic_filename[i] += ".fz"

            # Copy the image to the images directory. Note: the individual sector images that were
            # used to make the mosaic are left in place, as they will be needed if the mosaic
            # operation is reset without reseting the preceding image operation as well
            dst_dir = os.path.join(
                self.field.parset["dir_working"], "images", f"image_{self.index}"
            )
            os.makedirs(dst_dir, exist_ok=True)
            if self.skip_processing:
                # Single imaging sector: split on the sector name
                suffix = self.mosaic_filename[i].split("sector_1")[-1]
            else:
                # Mosacking done: split on the mosaic name
                suffix = self.mosaic_filename[i].split(self.name)[-1]
            field_image_filename = os.path.join(dst_dir, f"field{suffix}")
            if image_name == "I_image_file_true_sky":
                # Save the Stokes I true-sky image filename as an attribute of the field
                # object for later use
                self.field.field_image_filename_prev = self.field.field_image_filename
                self.field.field_image_filename = field_image_filename
            src_filename = os.path.join(self.pipeline_working_dir, self.mosaic_filename[i])
            if os.path.exists(src_filename):
                shutil.copy(src_filename, field_image_filename)

        # Finally call finalize() in the parent class
        super().finalize()
