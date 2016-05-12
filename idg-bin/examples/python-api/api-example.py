#!/usr/bin/env python

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import IDG

######
# main
######
def main():

    bufferTimesteps = 100

    plan = IDG.GridderPlan(bufferTimesteps)
