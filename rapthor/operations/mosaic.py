"""
Module that holds the Mosaic class
"""
import os
import logging
import shutil
from rapthor.lib.operation import Operation
from rapthor.lib.cwl import CWLFile
from rapthor.lib import miscellaneous as misc

log = logging.getLogger('rapthor:mosaic')


class Mosaic(Operation):
    """
    Operation to mosaic sector images
    """
    def __init__(self, field, index):
        super().__init__(field, index=index, name='mosaic')

        # For each image type we use a subworkflow, so we set the template filename
        # for that here
        self.subpipeline_parset_template = '{0}_type_pipeline.cwl'.format(self.rootname)

        # Determine whether processing is needed
        self.skip_processing = len(self.field.imaging_sectors) < 2

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        if self.batch_system.startswith('slurm'):
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.field.parset['cluster_specific']['max_cores']
        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'pipeline_working_dir': self.pipeline_working_dir,
                             'max_cores': max_cores,
                             'skip_processing': self.skip_processing,
                             'compress_images': self.field.compress_images}

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        # Define various input and output filenames
        sector_image_filename = []
        sector_vertices_filename = []
        regridded_image_filename = []
        template_image_filename = []
        self.image_names = []  # list of input image names
        for pol in self.field.image_pol:
            polup = pol.upper()
            self.image_names.extend([f'{polup}_image_file_true_sky',
                                     f'{polup}_image_file_apparent_sky',
                                     f'{polup}_model_file_true_sky',
                                     f'{polup}_residual_file_apparent_sky'])
            if self.field.save_supplementary_images:
                self.image_names.append(f'{polup}_dirty_file_apparent_sky')
                if 'mask_filename' not in self.image_names:
                    self.image_names.append('mask_filename')

        for image_name in self.image_names:
            image_list = []
            vertices_list = []
            regridded_list = []
            for sector in self.field.imaging_sectors:
                image_list.append(getattr(sector, image_name))
                vertices_list.append(sector.vertices_file)
                regridded_list.append(os.path.basename(getattr(sector, image_name)) + '.regridded')
            sector_image_filename.append(CWLFile(image_list).to_json())
            sector_vertices_filename.append(CWLFile(vertices_list).to_json())
            regridded_image_filename.append(regridded_list)
            template_image_filename.append(self.name + '_template.fits')

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
                suffix = getattr(self.field.imaging_sectors[0], image_name).split('MFS')[-1]
                self.mosaic_filename.append('{0}-MFS{1}'.format(self.name, suffix))

        self.input_parms = {'skip_processing': self.skip_processing,
                            'sector_image_filename': sector_image_filename,
                            'sector_vertices_filename': sector_vertices_filename,
                            'template_image_filename': template_image_filename,
                            'regridded_image_filename': regridded_image_filename,
                            'mosaic_filename': self.mosaic_filename}

    def finalize(self):
        """
        Finalize this operation
        """
        for i, image_name in enumerate(self.image_names):
            if self.mosaic_filename[i] is None:
                continue

            # Copy the image to the images directory
            dst_dir = os.path.join(self.field.parset['dir_working'], 'images',
                                   'image_{}'.format(self.index))
            os.makedirs(dst_dir, exist_ok=True)
            suffix = getattr(self.field.imaging_sectors[0], image_name).split('MFS')[-1]
            field_image_filename = os.path.join(dst_dir, 'field-MFS{}'.format(suffix))
            if image_name == 'I_image_file_true_sky':
                # Save the Stokes I true-sky image filename as an attribute of the field
                # object for later use
                self.field.field_image_filename_prev = self.field.field_image_filename
                self.field.field_image_filename = field_image_filename
            shutil.copy(os.path.join(self.pipeline_working_dir, self.mosaic_filename[i]),
                        field_image_filename)

        # Finally call finalize() in the parent class
        super().finalize()
