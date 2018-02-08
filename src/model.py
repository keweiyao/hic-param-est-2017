"""
Computes model observables to match experimental data.
Prints all model data when run as a script.

Model data files are expected with the file structure
:file:`model_output/{design}/{system}/{design_point}.dat`, where
:file:`{design}` is a design type, :file:`{system}` is a system string, and
:file:`{design_point}` is a design point name.

For example, the structure of my :file:`model_output` directory is ::

    model_output
    ├── main
    │   ├── PbPb2760
    │   │   ├── 000.dat
    │   │   └── 001.dat
    │   └── PbPb5020
    │       ├── 000.dat
    │       └── 001.dat
    └── validation
        ├── PbPb2760
        │   ├── 000.dat
        │   └── 001.dat
        └── PbPb5020
            ├── 000.dat
            └── 001.dat

I have two design types (main and validation), two systems, and my design
points are numbered 000-499 (most numbers omitted for brevity).

Data files are expected to have the binary format created by my `heavy-ion
collision event generator
<https://github.com/jbernhard/heavy-ion-collisions-osg>`_.

Of course, if you have a different data organization scheme and/or format,
that's fine.  Modify the code for your needs.
"""

import logging
from pathlib import Path
import pickle
import subprocess
import h5py

from hic import flow
import numpy as np
from sklearn.externals import joblib

from . import workdir, cachedir, systems, lazydict, expt
from .design import Design

def _data(system, dataset='main'):
    print(system)
    """
    Compute model observables for the given system and dataset.

    dataset may be one of:

        - 'main' (training design)
        - 'validation' (validation design)
        - 'map' (maximum a posteriori, i.e. "best-fit" point)

    """
    if dataset not in {'main', 'validation', 'map'}:
        raise ValueError('invalid dataset: {}'.format(dataset))

    filep = Path(workdir, 'model_output', dataset, system, 'hybrid.hdf5')

    cachefile = Path(cachedir, 'model', dataset, system, 'hybrid.hdf5')

    logging.info(
        'loading %s/%s data and computing observables',
        system, dataset
    )

    expdata = expt.data[system]
    raax_0_10 = expdata['RAA']['D0']['0-10']['x']
    raax_0_100 = expdata['RAA']['D0']['0-100']['x']
    v2x_0_10 = expdata['V2']['D0']['0-10']['x']
    v2x_10_30 = expdata['V2']['D0']['10-30']['x']
    v2x_30_50 = expdata['V2']['D0']['30-50']['x']
    f = h5py.File(filep, 'r')
    modeldata = {'EPS09':{}, 'nCTEQ':{}}
    xraa =1
    xv2=1 
    for nPDF in ['EPS09', 'nCTEQ']:
        Raa_0_10 = np.array([p[nPDF+'/Raa-pT/mean'][0] for p in f.values()])
        Raa_0_100 = np.array([p[nPDF+'/Raa-pT/mean'][1] for p in f.values()])
        v2_0_10 = np.array([p[nPDF+'/vn2-pT/mean'][0, 1:,0] for p in f.values()])
        v2_10_30 = np.array([p[nPDF+'/vn2-pT/mean'][1,:,0] for p in f.values()])
        v2_30_50 = np.array([p[nPDF+'/vn2-pT/mean'][2,:,0] for p in f.values()])
        modeldata[nPDF] = {    'RAA': {'D0': { '0-10': {'x': raax_0_10, 
                                                        'Y': Raa_0_10 }, 
                                               '0-100': {'x': raax_0_100, 
                                                         'Y': Raa_0_100}
                                            }
                                        
                                            },
                                'V2': {'D0': {'0-10': {'x': v2x_0_10, 
                                                       'Y': v2_0_10}, 
                                              '10-30':{'x': v2x_10_30,  
                                                       'Y': v2_10_30},
                                              '30-50':{'x': v2x_30_50, 
                                                       'Y': v2_30_50}
                                            }
                                      }            
                }



    

    return modeldata


data = {s: _data(s, 'main') for s in systems}

if __name__ == '__main__':
    from pprint import pprint
    for s in systems:
        d = data[s]
        print(s)
        pprint(d)
