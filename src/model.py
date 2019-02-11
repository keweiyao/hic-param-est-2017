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
	│   ├── Pb+Pb+2760
	│   │   ├── 000.dat
	│   │   └── 001.dat
	│   └── Pb+Pb+5020
	│  	 ├── 000.dat
	│  	 └── 001.dat
	└── validation
		├── Pb+Pb+2760
		│   ├── 000.dat
		│   └── 001.dat
		└── Pb+Pb+5020
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

	cachefile = Path(cachedir, 'model', dataset, system, 'obs.h5')
	print(cachefile)
	logging.info(
		'loading %s/%s data and computing observables',
		system, dataset
	)

	expdata = expt.data[system]
	raax_0_10 = expdata['CMS']['RAA']['D0']['0-10']['x']
	raax_0_100 = expdata['CMS']['RAA']['D0']['0-100']['x']
	v2x_0_10 = expdata['CMS']['V2']['D0']['0-10']['x']
	v2x_10_30 = expdata['CMS']['V2']['D0']['10-30']['x']
	v2x_30_50 = expdata['CMS']['V2']['D0']['30-50']['x']
	raaAx_0_10 = expdata['ALICE']['RAA']['D-avg']['0-10']['x']
	raaAx_30_50 = expdata['ALICE']['RAA']['D-avg']['30-50']['x']
	raaAx_60_80 = expdata['ALICE']['RAA']['D-avg']['60-80']['x']
	v2Ax_30_50 = expdata['ALICE']['V2']['D-avg']['30-50']['x']
	v2Aeex_30_50 = expdata['ALICE']['V2']['D-avg']['30-50-H']['x']

	def obs_dtype(centralities, NpTs):
		return [ 
					("{:s}".format(cen), 
						[ ('xbins', np.float, [NpT,2]),
						  ('x', np.float, NpT),
						  ('y', np.float, NpT),
						  ('yerr', np.float, NpT)
						]
					)
					for cen, NpT in zip(centralities, NpTs)
				]
	dtype=[
	('CMS', [ ('RAA', [
		('D0', obs_dtype(['0-10','0-100'], [raax_0_10.size, raax_0_100.size])	)
						   ]
				  ), 
	  ('V2',  [
					('D0', obs_dtype(['0-10','10-30','30-50'], [v2x_0_10.size,v2x_10_30.size,v2x_30_50.size])	)
			  ]
	  ), 
				]
		),
		('ALICE', [ ('RAA', [
								('D-avg', obs_dtype(['0-10','30-50','60-80'], [raaAx_0_10.size, raaAx_30_50.size, raaAx_60_80.size])	)
							]
					),
					('V2',  [
								('D-avg', obs_dtype(['30-50'], [v2Ax_30_50.size])	)
							]
					),	
				  ]
		),
		]
	
	if cachefile.exists():
		d = Design("Pb-Pb-5020", npoints=250 if dataset!="validation" else 50, validation = (dataset=="validation"))
		AllData = []
		with h5py.File(cachefile,'r') as f:
			for point in d.points:
				data = np.empty([], dtype=dtype)
				g = f['/{}.dat'.format(point)]
				# CMS Raa
				xbins = g['CMS/Raa/pT-bins'].value
				x = (xbins[:,0]+xbins[:,1])/2.
				y = g['CMS/Raa/D/y'].value
				yerr = g['CMS/Raa/D/yerr'].value
				data['CMS']['RAA']['D0']['0-10']['xbins'] = xbins
				data['CMS']['RAA']['D0']['0-100']['xbins'] = xbins
				data['CMS']['RAA']['D0']['0-10']['x'] = x
				data['CMS']['RAA']['D0']['0-100']['x'] = x
				data['CMS']['RAA']['D0']['0-10']['y'] = y[0]
				data['CMS']['RAA']['D0']['0-10']['yerr'] = yerr[0]
				data['CMS']['RAA']['D0']['0-100']['y'] = y[1]
				data['CMS']['RAA']['D0']['0-100']['yerr'] = yerr[1]	
				# CMS V2
				xbins = g['CMS/vn_HF/pT-bins'].value
				x = (xbins[:,0]+xbins[:,1])/2.
				y = g['CMS/vn_HF/D+D*/y'].value
				yerr = g['CMS/vn_HF/D+D*/yerr'].value
				data['CMS']['V2']['D0']['0-10']['xbins'] = xbins[1:]
				data['CMS']['V2']['D0']['10-30']['xbins'] = xbins
				data['CMS']['V2']['D0']['30-50']['xbins'] = xbins
				data['CMS']['V2']['D0']['0-10']['x'] = x[1:]
				data['CMS']['V2']['D0']['10-30']['x'] = x
				data['CMS']['V2']['D0']['30-50']['x'] = x
				data['CMS']['V2']['D0']['0-10']['y'] = y[0,1:,0]
				data['CMS']['V2']['D0']['10-30']['y'] = y[1,:,0]
				data['CMS']['V2']['D0']['30-50']['y'] = y[2,:,0]
				data['CMS']['V2']['D0']['0-10']['yerr'] = yerr[0,1:,0]
				data['CMS']['V2']['D0']['10-30']['yerr'] = yerr[1,:,0]
				data['CMS']['V2']['D0']['30-50']['yerr'] = yerr[2,:,0]		
				# ALICE Raa
				xbins = g['ALICE/Raa/pT-bins'].value
				x = (xbins[:,0]+xbins[:,1])/2.
				y = g['ALICE/Raa/D+D*/y'].value
				yerr = g['ALICE/Raa/D+D*/yerr'].value
				data['ALICE']['RAA']['D-avg']['0-10']['xbins'] = xbins
				data['ALICE']['RAA']['D-avg']['30-50']['xbins'] = xbins[:-1]
				data['ALICE']['RAA']['D-avg']['60-80']['xbins'] = xbins[:-1]
				data['ALICE']['RAA']['D-avg']['0-10']['x'] = x
				data['ALICE']['RAA']['D-avg']['30-50']['x'] = x[:-1]
				data['ALICE']['RAA']['D-avg']['60-80']['x'] = x[:-1]
				data['ALICE']['RAA']['D-avg']['0-10']['y'] = y[0,:]
				data['ALICE']['RAA']['D-avg']['30-50']['y'] = y[1,:-1]
				data['ALICE']['RAA']['D-avg']['60-80']['y'] = y[2,:-1]
				data['ALICE']['RAA']['D-avg']['0-10']['yerr'] = yerr[0,:]
				data['ALICE']['RAA']['D-avg']['30-50']['yerr'] = yerr[1,:-1]
				data['ALICE']['RAA']['D-avg']['60-80']['yerr'] = yerr[2,:-1]
				# ALICE V2
				xbins = g['ALICE/vn_HF/pT-bins'].value
				x = (xbins[:,0]+xbins[:,1])/2.
				y = g['ALICE/vn_HF/D+D*/y'].value
				yerr = g['ALICE/vn_HF/D+D*/yerr'].value
				data['ALICE']['V2']['D-avg']['30-50']['xbins'] = xbins
				data['ALICE']['V2']['D-avg']['30-50']['x'] = x
				data['ALICE']['V2']['D-avg']['30-50']['y'] = y[0,:,0]
				data['ALICE']['V2']['D-avg']['30-50']['yerr'] = yerr[0,:,0]
		
				AllData.append(data)
		AllData = np.array(AllData)
		return AllData



data = {s: _data(s, 'main') for s in systems}
data_validation = {s: _data(s, 'validation') for s in systems}

if __name__ == '__main__':
	from pprint import pprint
	for s in systems:
		d = data[s]
		print(s)
		pprint(d)
