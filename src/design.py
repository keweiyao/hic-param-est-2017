"""
Generates Latin-hypercube parameter designs.

When run as a script, writes input files for use with my
`heavy-ion collision event generator
<https://github.com/jbernhard/heavy-ion-collisions-osg>`_.
Run ``python -m src.design --help`` for usage information.

.. warning::

	This module uses the R `lhs package
	<https://cran.r-project.org/package=lhs>`_ to generate maximin
	Latin-hypercube samples.  As far as I know, there is no equivalent library
	for Python (I am aware of `pyDOE <https://pythonhosted.org/pyDOE>`_, but
	that uses a much more rudimentary algorithm for maximin sampling).

	This means that R must be installed with the lhs package (run
	``install.packages('lhs')`` in an R session).

"""

import itertools
import logging
from pathlib import Path
import re
import subprocess

import numpy as np

from . import cachedir, parse_system

def generate_lhs(npoints, ndim, seed):
	"""
	Generate a maximin Latin-hypercube sample (LHS) with the given number of
	points, dimensions, and random seed.

	"""
	logging.debug(
		'generating maximin LHS: '
		'npoints = %d, ndim = %d, seed = %d',
		npoints, ndim, seed
	)

	cachefile = (
		cachedir / 'lhs' /
		'npoints{}_ndim{}_seed{}.npy'.format(npoints, ndim, seed)
	)

	if cachefile.exists():
		logging.debug('loading from cache')
		lhs = np.load(cachefile)
	else:
		logging.debug('not found in cache, generating using R')
		proc = subprocess.run(
			['R', '--slave'],
			input="""
			library('lhs')
			set.seed({})
			write.table(maximinLHS({}, {}), col.names=FALSE, row.names=FALSE)
			""".format(seed, npoints, ndim).encode(),
			stdout=subprocess.PIPE,
			check=True
		)

		lhs = np.array(
			[l.split() for l in proc.stdout.splitlines()],
			dtype=float
		)

		cachefile.parent.mkdir(exist_ok=True)
		np.save(cachefile, lhs)

	return lhs


class Design:
	"""
	Latin-hypercube model design.

	Creates a design for the given system with the given number of points.
	Creates the main (training) design if `validation` is false (default);
	creates the validation design if `validation` is true.  If `seed` is not
	given, a default random seed is used (different defaults for the main and
	validation designs).

	Public attributes:

	- ``system``: the system string
	- ``projectiles``, ``beam_energy``: system projectile pair and beam energy
	- ``type``: 'main' or 'validation'
	- ``keys``: list of parameter keys
	- ``labels``: list of parameter display labels (for TeX / matplotlib)
	- ``range``: list of parameter (min, max) tuples
	- ``min``, ``max``: numpy arrays of parameter min and max
	- ``ndim``: number of parameters (i.e. dimensions)
	- ``points``: list of design point names (formatted numbers)
	- ``array``: the actual design array

	The class also implicitly converts to a numpy array.

	This is probably the worst class in this project, and certainly the least
	generic.  It will probably need to be heavily edited for use in any other
	project, if not completely rewritten.

	"""
	def transform(self, samples, Texp=False):
		if Texp:
			samples[1] = np.exp(samples[1])
			samples[2] = np.exp(samples[2])
			samples[3] = np.exp(samples[3])
			samples[7] = np.exp(samples[7])
			samples[8] = np.exp(samples[8])
			labels = [r'$\tau_i$', r'$c$', r'$R_v$', r'$\mu$', r'$K$',r'$p$',  r'$q$',  r'$a$',  r'$b$',  r'$\gamma$', r'$\sigma_m$']
		else: 
			labels = self.labels + [r'$\sigma_m$']
		samples[4] *= 5.0 # reparametrized K, differ from the orginal K by a factor of 5
		ranges = np.array([np.min(samples, axis=1), np.max(samples, axis=1)]).T
		return labels, ranges, samples

	def __init__(self, system, npoints=250, validation=False, seed=None):
		self.npoints = npoints
		self.system = system
		self.proj, self.targ, self.beam_energy = parse_system(system)
		self.type = 'validation' if validation else 'main'

		self.keys, self.labels, self.range = map(list, zip(*[
			('xi' ,  r'$\tau_i/\tau_0$',	   (.1, .9)	),
			('lnc' ,  r'$\ln(c)$',	   (np.log(1.), np.log(10.))	),
			('lnRv',  r'$\ln(R_{v})$',(np.log(1.), np.log(8.))		),
			('lnmu',  r'$\ln(\mu)$',   (np.log(.6), np.log(10.0))	),
			('K',      r'$K$',         (0,4)		),
			('p'	,  r'$p$',         (-2.,2.)		),
			('q'	,  r'$q$',         (-1.,1.)	),
			('lna'	,  r'$\ln(a)$',    (np.log(.5),np.log(3))		),
			('lnb'	,  r'$\ln(b)$',    (np.log(.5),np.log(3)) 		),
			('gamma',  r'$\gamma$',    (-1,1)	),

		]))

		self.ndim = len(self.range)
		self.min, self.max = map(np.array, zip(*self.range))

		# use padded numbers for design point names
		fmt = '{:0' + str(len(str(npoints - 1))) + 'd}'
		self.points = [fmt.format(i) for i in range(npoints)]

		if seed is None:
			seed = 751783496 if validation else 450829120

		self.array = self.min + (self.max - self.min)*generate_lhs(
			npoints=npoints, ndim=self.ndim, seed=seed
		)
		
		self._template =\
"""proj = {proj}
targ = {targ}
NPythiaEvents = {Npythia}
sqrts = {sqrts}
lido-args =  -m {mu} -f -1. -c {cut} -r {Rv} -k {K} -a {a} -b {b} -p {p} -q {q} -g {g}
trento-args = -x {x} -n {norm} -p 0.0 -k 1.0 -w 0.6 --ncoll
tau-fs = {fs}
xi-fs = {xi}
hydro-args = stop=0.150 min=0.08 slope=1.1 curvature=-0.5 zetas_max=0.05 zetas_width=0.02 zetas_t0=0.180 iskip_t=1 iskip_xy=2
Tswitch = 0.154
"""
	def __array__(self):
		return self.array
		
	def write_files(self, basedir):
		"""
		Write input files for each design point to `basedir`.

		"""
		outdir = basedir / self.type / self.system
		outdir.mkdir(parents=True, exist_ok=True)

		for point, row in zip(self.points, self.array):
			kwargs = {
				"proj": self.proj,
				"targ": self.targ,
				"Npythia": 50000,
				"sqrts": self.beam_energy,
				"x": {200: 4.2, 2760: 6.4, 5020: 7.0}[self.beam_energy],
				"norm": {200: 6.1, 2760: 13.9, 5020: 18.4}[self.beam_energy],
                "fs": {200: 0.5, 2760: 1.2, 5020: 1.2}[self.beam_energy],
				"xi": row[self.keys.index('xi')],
				"cut": np.exp(row[self.keys.index('lnc')]),
				"Rv": np.exp(row[self.keys.index('lnRv')]),
				"mu": np.exp(row[self.keys.index('lnmu')]),
				"K": row[self.keys.index('K')],
				"p": row[self.keys.index('p')],
				"q": row[self.keys.index('q')],
				"a": np.exp(row[self.keys.index('lna')]),
				"b": np.exp(row[self.keys.index('lnb')]),
				"g": row[self.keys.index('gamma')],
			}
			filepath = outdir / point
			with filepath.open('w') as f:
				f.write(self._template.format(**kwargs))
				logging.debug('wrote %s', filepath)

def main():
	import argparse
	from . import systems

	parser = argparse.ArgumentParser(description='generate design input files')
	parser.add_argument(
		'inputs_dir', type=Path,
		help='directory to place input files'
	)
	args = parser.parse_args()

	for system, validation in itertools.product(systems, [False, True]):
		Design(system, npoints=250 if not validation else 50, validation=validation).write_files(args.inputs_dir)

	logging.info('wrote all files to %s', args.inputs_dir)


if __name__ == '__main__':
	main()
