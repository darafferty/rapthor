"""
Definition of the master Operation class
"""
import os
import sys
import logging
from rapthor import _logging
from jinja2 import Environment, FileSystemLoader
from rapthor.lib import miscellaneous as misc
from toil.leader import FailedJobsException
from toil.cwl import cwltoil
import toil.version as toil_version
from rapthor.lib.context import Timer

DIR = os.path.dirname(os.path.abspath(__file__))
env_parset = Environment(loader=FileSystemLoader(os.path.join(DIR, '..', 'pipeline', 'parsets')))


class Operation(object):
    """
    Generic operation class

    An operation is simply a CWL pipeline that performs a part of the
    processing. It holds the pipeline settings, populates the pipeline input and
    parset templates, and runs the pipeline. The field object is passed between
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

        # Extra Toil env variables and Toil version
        self.toil_env_variables = {}
        self.toil_major_version = int(toil_version.version.split('.')[0])

        # rapthor working directory
        self.rapthor_working_dir = self.parset['dir_working']

        # Pipeline working dir
        self.pipeline_working_dir = os.path.join(self.rapthor_working_dir,
                                                 'pipelines', self.name)
        misc.create_directory(self.pipeline_working_dir)

        # Maximum number of nodes to use
        self.max_nodes = self.parset['cluster_specific']['max_nodes']

        # Directory that holds the pipeline logs in a convenient place
        self.log_dir = os.path.join(self.rapthor_working_dir, 'logs', self.name)
        misc.create_directory(self.log_dir)

        # Paths for scripts, etc. in the rapthor install directory
        self.rapthor_root_dir = os.path.split(DIR)[0]
        self.rapthor_pipeline_dir = os.path.join(self.rapthor_root_dir, 'pipeline')
        self.rapthor_script_dir = os.path.join(self.rapthor_root_dir, 'scripts')

        # Input template name and output parset and inputs filenames for
        # the pipeline. If the pipeline uses a subworkflow, its template filename must be
        # defined in the subclass by self.subpipeline_parset_template to the right
        # path
        self.pipeline_parset_template = '{0}_pipeline.cwl'.format(self.rootname)
        self.subpipeline_parset_template = None
        self.pipeline_parset_file = os.path.join(self.pipeline_working_dir,
                                                 'pipeline_parset.cwl')
        self.subpipeline_parset_file = os.path.join(self.pipeline_working_dir,
                                                    'subpipeline_parset.cwl')
        self.pipeline_inputs_file = os.path.join(self.pipeline_working_dir,
                                                 'pipeline_inputs.yml')

        # MPI configuration file
        self.mpi_config_file = os.path.join(self.pipeline_working_dir,
                                            'mpi_config.yml')

        # Toil's jobstore path
        self.jobstore = os.path.join(self.pipeline_working_dir, 'jobstore')

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

    def set_parset_parameters(self):
        """
        Define parameters needed for the pipeline parset template

        The dictionary keys must match the jinja template variables used in the
        corresponding pipeline parset.

        The entries are defined in the subclasses as needed
        """
        self.parset_parms = {}

    def set_input_parameters(self):
        """
        Define parameters needed for the pipeline inputs

        The dictionary keys must match the workflow inputs defined in the corresponding
        pipeline parset.

        The entries are defined in the subclasses as needed
        """
        self.input_parms = {}

    def setup(self):
        """
        Set up this operation

        This involves filling the pipeline parset template and writing the inputs file
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

        # Save the pipeline inputs to a file
        self.set_input_parameters()
        keys = []
        vals = []
        for k, v in self.input_parms.items():
            keys.append(k)
            if type(v) is bool:
                vals.append("'{}'".format(v))
            elif type(v) is list and type(v[0]) is bool:
                vals.append('[{}]'.format(','.join(["'{}'".format(ve) for ve in v])))
            else:
                vals.append(v)
        tmp = '\n'.join(['{0}: {1}'.format(k, v) for k, v in zip(keys, vals)])
        with open(self.pipeline_inputs_file, 'w') as f:
            f.write(tmp)

    def finalize(self):
        """
        Finalize this operation

        This should be defined in the subclasses as needed
        """
        pass

    def call_toil(self):
        """
        Calls Toil to run the operation's pipeline
        """
        # Build the args list
        args = []
        args.extend(['--singularity'])
        args.extend(['--batchSystem', self.batch_system])
        if self.batch_system == 'slurm':
            args.extend(['--disableCaching'])
            args.extend(['--defaultCores', str(self.cpus_per_task)])
            args.extend(['--defaultMemory', self.mem_per_node_gb])
            self.toil_env_variables['TOIL_SLURM_ARGS'] = "--export=ALL"
        args.extend(['--maxLocalJobs', str(self.max_nodes)])
        args.extend(['--jobStore', self.jobstore])
        if os.path.exists(self.jobstore):
            args.extend(['--restart'])
        args.extend(['--basedir', self.pipeline_working_dir])
        args.extend(['--outdir', self.pipeline_working_dir])
        args.extend(['--writeLogs', self.log_dir])
#        args.extend(['--logLevel', 'DEBUG'])  # used for debugging purposes only
        args.extend(['--maxLogFileSize', '0'])  # disable truncation of log files
        args.extend(['--preserve-entire-environment'])
#         args.extend(['--preserve-environment', 'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH'])
        if self.scratch_dir is not None:
            # Note: the trailing '/' is expected by Toil v5.3+
            args.extend(['--tmpdir-prefix', self.scratch_dir+'/'])
            args.extend(['--tmp-outdir-prefix', self.scratch_dir+'/'])
            args.extend(['--workDir', self.scratch_dir+'/'])
        args.extend(['--clean', 'never'])  # preserves the job store for future runs
#        args.extend(['--cleanWorkDir', 'never'])  # used for debugging purposes only
        args.extend(['--servicePollingInterval', '10'])
        args.extend(['--stats'])
        if self.field.use_mpi and self.toil_major_version >= 5:
            # Create the config file for MPI jobs and add the required args
            if self.batch_system == 'slurm':
                # Use salloc to request the SLRUM allocation and run the MPI job
                config_lines = ["runner: 'mpi_runner.sh'", "nproc_flag: '-N'",
                                "extra_flags: ['mpirun', '--map-by node']"]
            else:
                config_lines = ["runner: 'mpirun'", "nproc_flag: '-np'",
                                "extra_flags: ['--map-by node']"]
                self.log.warning('MPI support for non-Slurm clusters is experimental. '
                                 'Please report any issues encountered.')
            with open(self.mpi_config_file, 'w') as f:
                f.write('\n'.join(config_lines))
            args.extend(['--mpi-config-file', self.mpi_config_file])
            args.extend(['--enable-ext'])
        args.append(self.pipeline_parset_file)
        args.append(self.pipeline_inputs_file)

        # Set env variables, if any
        for k, v in self.toil_env_variables.items():
            os.environ[k] = v

        # Run the pipeline
        try:
            status = cwltoil.main(args=args)
            if status == 0:
                self.success = True
            else:
                self.success = False
        except FailedJobsException:
            self.success = False

        # Unset env variables, if any
        for k, v in self.toil_env_variables.items():
            os.environ[k] = ''

        # Reset the logging level, as the cwltoil call above can change it
        _logging.set_level(self.parset['logging_level'])

    def run(self):
        """
        Runs the operation
        """
        # Set up pipeline and call Toil
        self.setup()
        self.log.info('<-- Operation {0} started'.format(self.name))
        with Timer(self.log):
            self.call_toil()

        # Finalize
        if self.success:
            self.log.info('--> Operation {0} completed'.format(self.name))
            self.finalize()
        else:
            self.log.error('Operation {0} failed due to an error'.format(self.name))
            sys.exit(1)
