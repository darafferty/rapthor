from datetime import datetime

import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def add_key(indict, inkey, newkey, newval):
    ''' Simple function to add or update a dictionary key.

    Args:
        indict : dict
            Input dictionary to update.
        inkey : str
            Entry to which to add or update the key given by newkey.
        newkey : str
            The key to add.
        newval : object
            The value associated with newkey.
    '''
    try:
        indict[inkey].update({newkey:newval})
    except KeyError:
        indict[inkey] = {newkey:newval}
    

def group_by_cycle(cycledict):
    ''' Simple function to add or update a dictionary key.

    Args:
        cycledict : dict
            The dictionary containing steps from the rapthor log as generated in this script.
    
    Returns:
        groupcycledict: dict
            The dictionary of operations grouped by their cycle.
    '''
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
        add_key(groupcycledict, 'cycle_{:d}'.format(i), 'calibrate_{:d}'.format(i), cycledict['calibrate_{:d}'.format(i)])
        add_key(groupcycledict, 'cycle_{:d}'.format(i), 'predict_{:d}'.format(i), cycledict['predict_{:d}'.format(i)])
        add_key(groupcycledict, 'cycle_{:d}'.format(i), 'image_{:d}'.format(i), cycledict['image_{:d}'.format(i)])
        add_key(groupcycledict, 'cycle_{:d}'.format(i), 'mosaic_{:d}'.format(i), cycledict['mosaic_{:d}'.format(i)])
    return groupcycledict

# Extract only the debug lines about timing from the log.
with open('rapthor_timing.txt', 'w') as outfile:
    subprocess.run(['grep', 'Time for operation', os.path.abspath(sys.argv[1])], stdout=outfile)

# Gather the total time spent in each operation for each cycle.
# Within a cycle each operation is summed if it occurs multiple times (caused by e.g. a restart).
operations = {}
with open('rapthor_timing.txt', 'r') as f:
    for line in f.readlines():
        split1 = line.split('-')
        operation = split1[4].strip().split(':')[1]
        timestr = split1[-1].split('operation:')[-1].strip()
        if 'day' in timestr:
            day = timestr.split(',')[0]
            dayhours = int(day.split(' ')[0]) * 24
            hourstr = timestr.split(',')[1].strip()
            time = datetime.strptime(hourstr, '%H:%M:%S.%f')
            dec_hour = dayhours + time.hour + time.minute / 60. + time.second / 3600.
        else:
            time = datetime.strptime(timestr, '%H:%M:%S.%f')
            dec_hour = time.hour + time.minute / 60. + time.second / 3600.
        if operation not in operations.keys():
            operations[operation] = dec_hour
        else:
            operations[operation] += dec_hour
    
sns.set_style("white", {'xtick.bottom': True, 'ytick.left': True})
cycledict = operations
df = pd.DataFrame(cycledict.items(), columns=('Operation', 'Duration'))
df['Cycle'] = ['Cycle {:d}'.format(int(x.split('_')[-1])) for x in df['Operation']]
df['Operation'] = [x.split('_')[0] for x in df['Operation']]

for cycle in cycledict.keys():
    h = sns.histplot(df, x='Cycle', hue='Operation', weights='Duration', multiple='dodge', discrete=True)
# There are 4 operations currently: calibrate, predict, image and mosaic.
for i in range(4):
    labels = ['{:.2f}'.format(t) if t else '' for t in h.containers[i].datavalues]
    h.bar_label(h.containers[i], labels=labels, fontsize=8)
h.set(xlabel='Self calibration cycle', ylabel='Duration [h]')
h.figure.savefig('rapthor_timing.svg', bbox_inches='tight', dpi=300)
h.figure.savefig('rapthor_timing.png', bbox_inches='tight', dpi=300)

# File is no longer needed.
os.remove('rapthor_timing.txt')
