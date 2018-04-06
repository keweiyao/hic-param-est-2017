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
    def __init__(self, system, npoints=80, validation=False, seed=None):
        self.system = system
        self.projectiles, self.beam_energy = parse_system(system)
        self.type = 'validation' if validation else 'main'

        self.keys, labels, self.range = map(list, zip(*[
            ('tau_0',        '$\tau_0$',        (0.1, 1.0)),
            ('mu',            '$\log\mu$',    (-1.1,    1.4)),
            ('qhat_A',        '$\log (1+A)$',    (0.01,    1.6)),
            ('qhat_B',        '$\log (1+B)$',    (0.01,    1.6)),
        ]))

        # convert labels into TeX:
        #   - wrap normal text with \mathrm{}
        #   - escape spaces
        #   - surround with $$
        self.labels = [ i for i in labels ]

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

    def __array__(self):
        return self.array

    _template = """nevents = 4
sqrts = {sqrts}
trento_args = -x {cross_section} -n {normalization} -p 0.0 -k 1.2 -w 0.9 --ncoll
tau_fs = 1.2
xi_fs = {xi_fs}
vishnew_args = stop=0.153 min=0.08 slope=1.1 curvature=-0.5 zetas_max=0.05 zetas_width=0.02 zetas_t0=0.180 iskip_t=4 iskip_xy=2
Tswitch = 0.154
N_charm = {N_charm}
N_bottom = {N_bottom}
Emax = 200.
pTmin = .5
pTmax = 130.5
mu = {mu}
A = {qhat_A}
B = {qhat_B}"""

    def write_files(self, basedir):
        """
        Write input files for each design point to `basedir`.

        """
        outdir = basedir / self.type / self.system
        outdir.mkdir(parents=True, exist_ok=True)

        for point, row in zip(self.points, self.array):
            tau0 = row[self.keys.index('tau_0')]
            logmu = row[self.keys.index('mu')]
            logqhatA = row[self.keys.index('qhat_A')]
            logqhatB = row[self.keys.index('qhat_B')]
            kwargs = {
                "sqrts": self.beam_energy,
                "N_charm": 40000,
				"N_bottom": 10000,
                "xi_fs": tau0/1.2,
                "mu": np.exp(logmu),
                "qhat_A": np.exp(logqhatA)-1.,
                "qhat_B": np.exp(logqhatB)-1.,
                "normalization": {
                    2760: 13.9,
                    5020: 18.4,
                }[self.beam_energy],
                "cross_section": {
                    # sqrt(s) [GeV] : sigma_NN [fm^2]
                    200: 4.2,
                    2760: 6.4,
                    5020: 7.0,
                }[self.beam_energy]
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
        Design(system, validation=validation).write_files(args.inputs_dir)

    logging.info('wrote all files to %s', args.inputs_dir)


if __name__ == '__main__':
    main()
