#!/usr/bin/env python3
from datetime import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
import glob
import os
import re
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGSIZE = (11.7, 8.3)


class MainLogParser():
    """ Class for parsing the main Rapthor log."""

    def __init__(self, rapthorlog: str) -> None:
        """ Class initialiser.

        Args:
            rapthorlog : the main Rapthor log file.
        """
        self.file = os.path.abspath(rapthorlog)
        self.operations = {}

    def add_key(self, indict: dict, inkey: str, newkey: str, newval: object) -> None:
        """ Simple function to add or update a dictionary key.

        Args:
            indict
                Input dictionary to update.
            inkey
                Entry to which to add or update the key given by newkey.
            newkey
                The key to add.
            newval
                The value associated with newkey.
        """
        try:
            indict[inkey].update({newkey: newval})
        except KeyError:
            indict[inkey] = {newkey: newval}

    def group_by_cycle(self, cycledict: dict) -> dict:
        """ Simple function to add or update a dictionary key.

        Args:
            cycledict : dict
                The dictionary containing steps from the rapthor log as generated in this script.

        Returns:
            groupcycledict: dict
                The dictionary of operations grouped by their cycle.
        """
        ncycles = 1
        for k in cycledict.keys():
            cycle = int(k.split('_')[-1])
            if cycle > ncycles:
                ncycles += 1
        cycle_keys = ['cycle_{:d}'.format(i) for i in range(1, ncycles)]
        print(cycle_keys)
        groupcycledict = {}
        # Assume operation names won't change
        # Rapthor does calibrate_N, predict_N, image_N, mosaic_N
        for i, k in enumerate(cycle_keys, start=1):
            self.add_key(groupcycledict, 'cycle_{:d}'.format(i), 'calibrate_{:d}'.format(i), cycledict['calibrate_{:d}'.format(i)])
            self.add_key(groupcycledict, 'cycle_{:d}'.format(i), 'predict_{:d}'.format(i), cycledict['predict_{:d}'.format(i)])
            self.add_key(groupcycledict, 'cycle_{:d}'.format(i), 'image_{:d}'.format(i), cycledict['image_{:d}'.format(i)])
            self.add_key(groupcycledict, 'cycle_{:d}'.format(i), 'mosaic_{:d}'.format(i), cycledict['mosaic_{:d}'.format(i)])
        return groupcycledict

    def process(self) -> None:
        """ Process the main log and produce an overview graph."""
        # Extract only the debug lines about timing from the log.
        with open('rapthor_timing.txt', 'w') as outfile:
            subprocess.run(['grep', 'Time for operation', self.file], stdout=outfile)

        # Gather the total time spent in each operation for each cycle.
        # Within a cycle each operation is summed if it occurs multiple times (caused by e.g. a restart).
        with open('rapthor_timing.txt', 'r') as f:
            # Regex to get rid of the ANSI colour codes that may be present from logging.
            # https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            for line in f.readlines():
                split1 = line.split('-')
                operation = split1[4].strip().split(':')[1]
                timestr = split1[-1].split('operation:')[-1].strip()
                timestr = ansi_escape.sub('', timestr)
                if 'day' in timestr:
                    day = timestr.split(',')[0]
                    dayhours = int(day.split(' ')[0]) * 24
                    hourstr = timestr.split(',')[1].strip()
                    time = datetime.strptime(hourstr, '%H:%M:%S.%f')
                    dec_hour = dayhours + time.hour + time.minute / 60. + time.second / 3600.
                else:
                    time = datetime.strptime(timestr, '%H:%M:%S.%f')
                    dec_hour = time.hour + time.minute / 60. + time.second / 3600.
                if operation not in self.operations.keys():
                    self.operations[operation] = dec_hour
                else:
                    self.operations[operation] += dec_hour
        # File is no longer needed.
        os.remove('rapthor_timing.txt')

    def plot(self) -> None:
        """ Create an overview plot. """
        sns.set_style("white", {'xtick.bottom': True, 'ytick.left': True})
        # Some operations are only done once and don't have a cycle count
        # suffix. Just add `_1` in this case to add them to the first cycle.
        cycledict = {
            key if key[-1].isdigit() else key + '_1': value
            for key, value in self.operations.items()
        }
        df = pd.DataFrame(cycledict.items(), columns=('Operation', 'Duration'))
        df['Cycle'] = ['Cycle {:d}'.format(int(x.split('_')[-1])) for x in df['Operation']]
        df['Operation'] = [x.rsplit('_', 1)[0] for x in df['Operation']]

        fig = plt.figure(figsize=FIGSIZE)
        h = sns.histplot(df, x='Cycle', hue='Operation', weights='Duration', multiple='dodge', discrete=True, figure=fig)
        # Check how many of the 4 operations (calibrate, predict, image and mosaic) are present.
        Nops = len(pd.unique(df['Operation']))
        for i in range(Nops):
            try:
                labels = ['{:.2f}'.format(t) if t else '' for t in h.containers[i].datavalues]
                h.bar_label(h.containers[i], labels=labels, fontsize=8)
            except AttributeError:
                print('Failed to set bar labels. Try updating matplotlib to 3.4 or newer.')
        h.set(xlabel='Self calibration cycle', ylabel='Duration [h]', title='Cumulative runtime: {:.2f} hours'.format(df['Duration'].sum()))
        h.figure.savefig('rapthor_timing.pdf', bbox_inches='tight', dpi=300)
        h.figure.savefig('rapthor_timing.png', bbox_inches='tight', dpi=300)


class SubLogParser():
    """ Class to parse the Toil logs of each separate step."""

    def __init__(self, logdir: str, operation: str) -> None:
        """ Initialise the sub log parser.

        Args:
            logdir : path to the logs directory of Rapthor.
            operation: the operation to be parsed (e.g. calibrate, image etc.)
        """
        self.operation = operation
        self.files = []
        self.sub_names = []
        self.run_times = []

        op_files = list(glob.glob(os.path.join(logdir, operation, 'CWLJob*.log')))
        if not op_files:
            raise ValueError('Could not find files named "CWLJob*.log". Pipeline was not run with Toil or all steps failed.')
        op_files.sort(key=lambda x: os.path.getmtime(x))
        for sublog in op_files:
            self.files.append(os.path.abspath(sublog))
            if 'subpipeline' in sublog:
                sub_name = sublog.split('/')[-1].replace('CWLJob_subpipeline_parset.cwl.', '').split('--')[0]
            elif 'pipeline' in sublog:
                sub_name = sublog.split('/')[-1].replace('CWLJob_pipeline_parset.cwl.', '').split('--')[0]
            sub_name = sub_name.replace('_kind', '')
            sub_name = sub_name.replace('.cwl', '')
            self.sub_names.append(sub_name)

    def get_run_times(self) -> list:
        """ Obtain the run time of each step in seconds."""
        if not self.run_times:
            times = []
            for f in self.files:
                proc = subprocess.Popen(['grep', 'we ran for a total of', f], stdout=subprocess.PIPE)
                output = proc.stdout.read().decode('utf-8').strip()
                # The string we are after here always ends with "a total of X seconds" when using Toil.
                times.append(float(output.split(' ')[-2]))
            return times
        else:
            return self.run_times

    def plot(self) -> None:
        """ Create an overview graph of all the sub steps."""
        sns.set_style("white", {'xtick.bottom': True, 'ytick.left': True})
        data = {'Subtask': self.sub_names, 'Runtime': self.get_run_times()}

        df = pd.DataFrame.from_dict(data, orient='columns')

        fig = plt.figure(figsize=FIGSIZE)
        if df['Runtime'].max() > 3600:
            h = sns.histplot(df, y='Subtask', weights='Runtime', discrete=True, figure=fig, log_scale=[True, False])
            h.set(xlabel='Duration [s]', ylabel=None, title=self.operation)
        else:
            h = sns.histplot(df, y='Subtask', weights='Runtime', discrete=True, figure=fig)
            h.set(xlabel='Duration [s]', ylabel=None, title=self.operation)
        h.figure.savefig('temp_{:s}.pdf'.format(self.operation), bbox_inches='tight', dpi=300)
        h.figure.savefig('temp_{:s}.png'.format(self.operation), bbox_inches='tight', dpi=300)


def make_cycle_pdfs_sublogs() -> None:
    """ Make summary PDFs of each processing cycle and as a grand total."""
    # Every cycle will do calibration, so determine the number of cycles from this.
    Ncycles = len(glob.glob('temp_calibrate_*.pdf'))
    try:
        for cycle in range(1, Ncycles+1):
            print(f'Attempting to concat PDF files for cycle {cycle} with pdfunite')
            files = glob.glob(f'temp_*{cycle}.pdf')
            for f in files:
                if not os.path.isfile(os.path.abspath(f)):
                    files.remove(f)
            cmd = ['pdfunite'] + files + [f'summary_cycle_{cycle}.pdf']
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        cmd = ['pdfunite', 'rapthor_timing.pdf'] + sorted(glob.glob('summary_cycle_*.pdf')) + ['summary.pdf']
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # If we were successful, clean up the temporary plots.
        for f in glob.glob('temp_*.pdf'):
            os.remove(f)
            os.remove(f.replace('pdf', 'png'))
    except Exception:
        print('Concatenation failed. Is pdfunite installed?')


def main(logdir, detailed: bool = False) -> None:
    """ Main entry point."""
    main_log_file = os.path.join(os.path.abspath(logdir), 'rapthor.log')
    main_log = MainLogParser(main_log_file)
    main_log.process()
    main_log.plot()

    if detailed:
        print('Found the following operations:', ', '.join(main_log.operations.keys()))
        for op in main_log.operations.keys():
            print('Processing logs for operation {:s}'.format(op))
            try:
                sub = SubLogParser(os.path.abspath(logdir), op)
                sub.plot()
            except ValueError:
                print('No appropriate log files found for {:s}; skipping.'.format(op))
        make_cycle_pdfs_sublogs()


if __name__ == '__main__':
    descriptiontext = "Produce a break down of the rapthor logs. "

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('logdir', help='Directory where the logs are located.')
    parser.add_argument('--detailed', help='Produce a detailed overview by breaking down each operation in its substeps. Requires Toil as the chosen CWL runner.', action='store_true', required=False)
    args = parser.parse_args()
    main(args.logdir, args.detailed)
