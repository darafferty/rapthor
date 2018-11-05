# SKA1 Layout files are taken from layouts found in the SKA ECP register:
# https://skaoffice.atlassian.net/wiki/display/EP/ECP+Register
# ECPs 140021, 140022 and 140023
#
# Layouts were converted from lon, lat using CASA Version 4.1.0 (r24668)
# with the following script.
#

def convert_layout(file_in, file_out):

    import numpy as np

    layout = np.genfromtxt(file_in, dtype=np.double, delimiter=',')

    layout_itrf = np.zeros((len(layout),3), dtype=np.double)

    for i in range(0, len(layout)):
        lon = layout[i,0]
        lat = layout[i,1]

        t = me.position('WGS84',
               qa.quantity(layout[i,0],'deg'),
               qa.quantity(layout[i,1],'deg'),
               qa.quantity(0,'m'))
        t = me.measure(t, 'ITRF')
        t = me.expand(t)
        layout_itrf[i,0] = t[1]['value'][0]
        layout_itrf[i,1] = t[1]['value'][1]
        layout_itrf[i,2] = t[1]['value'][2]

    np.savetxt(file_out, layout_itrf, fmt='%.3f', delimiter=',')



