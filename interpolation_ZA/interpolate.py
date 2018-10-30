#!/usr/bin/env python 

import glob
import os
import re
import math
import socket
import json

import array
import numpy as np
import argparse 

from useful_functions import EvaluationGrid

def get_options():
    """                                         
    Parse and return the arguments provided by the user.
    """
    parser = argparse.ArgumentParser(description='Compare names of input and output files for matches and potentiel failed files')

    #parser.add_argument('-p','--path', action='store', required=True, type=str,
    #                    help='Path of the files to be processed')
    #parser.add_argument('-o','--output', action='store', required=True, type=str,
    #                    help='Output files')
    #parser.add_argument('--part_name', action='store', required=False, default="", type=str,
    #                    help='Common part of the file names [DEFAULT = FALSE]')
    #parser.add_argument('--not_in_name', action='store', required=False, default=None, type=str,
    #                    help='Excluded part of name (use to exclude files) [DEFAULT = FALSE]')
    #parser.add_argument('--verbose', action='store', required=False, default=False, type=bool,
    #                    help='Verbosity : wether to look at sizes and number of events inside each files [DEFAULT = FALSE]')

    options = parser.parse_args()




def main():
    
    # Get options from user #
    opt = get_options()

    # Get grid on which evaluate the network for the interpolation #
    grid = EvaluationGrid() 
    print (grid)


if __name__ == "__main__":                                     
    main()

