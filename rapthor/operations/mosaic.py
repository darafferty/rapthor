"""
Module that holds the Mosaic class
"""
import os
import logging
from rapthor.lib.operation import Operation
from rapthor.lib import miscellaneous as misc

log = logging.getLogger('rapthor:mosaic')


class Mosaic(Operation):
    """
    Operation to mosaic sector images
    """
    def __init__(self, field, index):
        super(Mosaic, self).__init__(field, name='mosaic', index=index)

    def set_parset_parameters(self):
        """
        Define parameters needed for the pipeline parset template
        """
        if self.batch_system == 'slurm':
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.field.parset['cluster_specific']['max_cores']
        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'max_cores': max_cores,
                             'max_threads': self.field.parset['cluster_specific']['max_threads'],
                             'do_slowgain_solve': self.field.do_slowgain_solve}

    def set_input_parameters(self):
        """
        Define the pipeline inputs
        """
        # First, determine whether processing is needed
        if len(self.field.imaging_sectors) > 1:
            skip_processing = False
        else:
            # No need to mosaic if we have just one sector
            skip_processing = True

        # Define various input and output filenames
        sector_image_filename = []
        sector_vertices_filename = []
        regridded_image_filename = []
        for sector in self.field.imaging_sectors:
            sector_image_filename.append(sector.I_image_file_true_sky)
            sector_vertices_filename.append(sector.vertices_file)
            regridded_image_filename.append(sector.I_image_file_true_sky+'.regridded')
        self.mosaic_root = os.path.join(self.pipeline_working_dir, self.name)
        template_image_filename = self.mosaic_root + '_template.fits'
        self.mosaic_filename = self.mosaic_root + '-MFS-I-image.fits'

        self.input_parms = {'skip_processing': skip_processing,
                            'sector_image_filename': sector_image_filename,
                            'sector_vertices_filename': sector_vertices_filename,
                            'template_image_filename': template_image_filename,
                            'regridded_image_filename': regridded_image_filename,
                            'mosaic_filename': self.mosaic_filename}

    def finalize(self):
        """
        Finalize this operation
        """
        # Save the FITS image and model
        dst_dir = os.path.join(self.field.parset['dir_working'], 'images',
                               'image_{}'.format(self.index))
        misc.create_directory(dst_dir)
        self.field.field_image_filename = os.path.join(dst_dir, 'field-MFS-I-image.fits')
        os.system('cp {0} {1}'.format(self.mosaic_filename, self.field.field_image_filename))

        # TODO: make mosaic of model + QUV?
#         self.field_model_filename = os.path.join(dst_dir, 'field-MFS-I-model.fits')

        # TODO: clean up template+regridded images

