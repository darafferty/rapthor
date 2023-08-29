"""
Definition of the master Operation class
"""
import os
import sys
import logging
import json
from jinja2 import Environment, FileSystemLoader
from rapthor.lib import miscellaneous as misc
import toil.version as toil_version
from rapthor.lib.context import Timer
from rapthor.lib.cwl import NpEncoder
from rapthor.lib.cwlrunner import create_cwl_runner

DIR = os.path.dirname(os.path.abspath(__file__))
env_parset = Environment(loader=FileSystemLoader(os.path.join(DIR, '..', 'pipeline', 'parsets')))


class Operation(object):
    """
    Generic operation class

    An operation is simply a CWL workflow that performs a part of the
    processing. It holds the workflow settings, populates the workflow input and
    parset templates, and runs the workflow. The field object is passed between
    operations, each of which updates it with variables needed by other, subsequent,
    operations.

    Parameters
    ----------
    field : Field object
        Field for this operation
    name : str, optional
        Name of the operation
    index : int, optional
        Index of the operation
    """
    def __init__(self, field, name=None, index=None):
        self.parset = field.parset.copy()
        self.field = field
        self.rootname = name.lower()
        self.index = index
        if self.index is not None:
            self.name = '{0}_{1}'.format(self.rootname, self.index)
        else:
            self.name = self.rootname
        self.rootname
        self.parset['op_name'] = name
        self.log = logging.getLogger('rapthor:{0}'.format(self.name))
        self.force_serial_jobs = False  # force jobs to run serially
        self.use_mpi = False

        # Extra Toil env variables and Toil version
        self.toil_env_variables = {}
        self.toil_major_version = int(toil_version.version.split('.')[0])

        # Rapthor working directory
        self.rapthor_working_dir = self.parset['dir_working']

        # Workflow working dir
        self.pipeline_working_dir = os.path.join(self.rapthor_working_dir,
                                                 'pipelines', self.name)
        misc.create_directory(self.pipeline_working_dir)

        # CWL runner settings
        self.cwl_runner = self.parset['cluster_specific']['cwl_runner']
        self.debug_workflow = self.parset['cluster_specific']['debug_workflow']

        # Maximum number of nodes to use
        self.max_nodes = self.parset['cluster_specific']['max_nodes']

        # Directory that holds the workflow logs in a convenient place
        self.log_dir = os.path.join(self.rapthor_working_dir, 'logs', self.name)
        misc.create_directory(self.log_dir)

        # Paths for scripts, etc. in the rapthor install directory
        self.rapthor_root_dir = os.path.split(DIR)[0]
        self.rapthor_pipeline_dir = os.path.join(self.rapthor_root_dir, 'pipeline')
        self.rapthor_script_dir = os.path.join(self.rapthor_root_dir, 'scripts')

        # Input template name and output parset and inputs filenames for the CWL workflow.
        # If the workflow uses a subworkflow, its template filename must be defined in the
        # subclass by self.subpipeline_parset_template to the right path
        self.pipeline_parset_template = '{0}_pipeline.cwl'.format(self.rootname)
        self.subpipeline_parset_template = None
        self.pipeline_parset_file = os.path.join(self.pipeline_working_dir,
                                                 'pipeline_parset.cwl')
        self.subpipeline_parset_file = os.path.join(self.pipeline_working_dir,
                                                    'subpipeline_parset.cwl')
        self.pipeline_inputs_file = os.path.join(self.pipeline_working_dir,
                                                 'pipeline_inputs.json')
        self.pipeline_outputs_file = os.path.join(self.pipeline_working_dir,
                                                  'pipeline_outputs.json')
        self.pipeline_log_file = os.path.join(self.log_dir, 'pipeline.log')

        # MPI configuration file
        self.mpi_config_file = os.path.join(self.pipeline_working_dir,
                                            'mpi_config.yml')

        # Toil's jobstore path
        self.jobstore = os.path.join(self.pipeline_working_dir, 'jobstore')

        # File indicating whether a step was completely done.
        self.done_file = os.path.join(self.pipeline_working_dir, '.done')

        # Get the batch system to use
        self.batch_system = self.parset['cluster_specific']['batch_system']

        # Get the maximum number of nodes to use
        if self.force_serial_jobs or self.batch_system == 'single_machine':
            self.max_nodes = 1
        else:
            self.max_nodes = self.parset['cluster_specific']['max_nodes']

        # Get the number of processors per task (SLRUM only). This is passed to sbatch's
        # --cpus-per-task option (see https://slurm.schedmd.com/sbatch.html). By setting
        # this value to the number of processors per node, one can ensure that each
        # task gets the entire node to itself
        self.cpus_per_task = self.parset['cluster_specific']['cpus_per_task']

        # Get the amount of memory in GB per node (SLRUM only). This is passed to
        # sbatch's --mem option (see https://slurm.schedmd.com/sbatch.html)
        if self.parset['cluster_specific']['mem_per_node_gb'] == 0:
            # Handle special case in which all memory is requested (must be passed to
            # slurm as '0', without units)
            self.mem_per_node_gb = '0'
        else:
            self.mem_per_node_gb = '{}G'.format(self.parset['cluster_specific']['mem_per_node_gb'])

        # Set the temp directory local to each node
        self.scratch_dir = self.parset['cluster_specific']['dir_local']

        # Toil's coordination directory
        self.coordination_dir  = self.parset['cluster_specific']['dir_coordination']

        # Get the container type
        if self.parset['cluster_specific']['use_container']:
            self.container = self.parset['cluster_specific']['container_type']
        else:
            self.container = None

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template

        The dictionary keys must match the jinja template variables used in the
        corresponding workflow parset.

        The entries are defined in the subclasses as needed
        """
        self.parset_parms = {}

    def set_input_parameters(self):
        """
        Define parameters needed for the CWL workflow inputs

        The dictionary keys must match the workflow inputs defined in the corresponding
        workflow parset.

        The entries are defined in the subclasses as needed
        """
        self.input_parms = {}

    def setup(self):
        """
        Set up this operation

        This involves filling the workflow template and writing the inputs file
        """
        # Fill the parset template and save to a file
        self.set_parset_parameters()
        self.pipeline_parset_template = env_parset.get_template(self.pipeline_parset_template)
        tmp = self.pipeline_parset_template.render(self.parset_parms)
        with open(self.pipeline_parset_file, 'w') as f:
            f.write(tmp)
        if self.subpipeline_parset_template is not None:
            self.subpipeline_parset_template = env_parset.get_template(self.subpipeline_parset_template)
            tmp = self.subpipeline_parset_template.render(self.parset_parms)
            with open(self.subpipeline_parset_file, 'w') as f:
                f.write(tmp)

        # Save the workflow inputs to a file
        self.set_input_parameters()
        with open(self.pipeline_inputs_file, 'w') as f:
            f.write(json.dumps(self.input_parms, cls=NpEncoder, indent=4, sort_keys=True))

    def finalize(self):
        """
        Finalize this operation.

        Create a "done" file to indicate that this operations is done.
        Specializations should be defined in the subclasses as needed.
        """
        open(self.done_file, "w").close()

    def is_done(self):
        """
        Check if this operation is done, by checking if a "done" file exists.
        """
        return os.path.isfile(self.done_file)

    def run(self):
        """
        Runs the operation
        """
        # Set up CWL workflow and call CWL runner
        self.setup()
        self.log.info('<-- Operation {0} started'.format(self.name))

        # Run current operation only if it hasn't run already.
        success = self.is_done()
        if not success:
            with Timer(self.log):
                with create_cwl_runner(self.cwl_runner, self) as runner:
                    success = runner.run()

        # Finalize
        if success:
            self.log.info('--> Operation {0} completed'.format(self.name))
            self.finalize()
        else:
            raise RuntimeError('Operation {0} failed due to an error'.format(self.name))
