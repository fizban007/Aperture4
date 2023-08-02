import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rc("font", family="serif")
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import sys
sys.path.append('.')
from datalib import Data, flag_to_species

#file_path = 'Aperture4/bin/Data/fld.00000.h5';
#h5file = h5py.File(file_path, 'r')

data = Data("../bin/Data/")
print(data.conf)
print(data.tracked_ptc_id)


