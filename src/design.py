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
    def __init__(self, system, npoints=100, validation=False, seed=None):
        self.system = system
        self.projectiles, self.beam_energy = parse_system(system)
        self.type = 'validation' if validation else 'main'

        self.keys, labels, self.range = map(list, zip(*[
            ('qA',        '$qA$',    (1,    30.)),
            ('qB',        '$qB$',    (1,    30.)),
            ('u1',        '$u1$',    (0.0,    1.0)),
            ('u2',        '$u2$',    (0.0,    1.0)),
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

    _template = """q1 = {q1}
q2 = {q2}
q3 = {q3}
q4 = {q4}"""

    def write_files(self, basedir):
        """
        Write input files for each design point to `basedir`.

        """
        outdir = basedir / self.type / self.system
        outdir.mkdir(parents=True, exist_ok=True)

        for point, row in zip(self.points, self.array):
            q1 = row[self.keys.index('qA')]
            q4 = row[self.keys.index('qB')]
            u1 = row[self.keys.index('u1')]
            u2 = row[self.keys.index('u2')]
            x2 = (1-u2)/(1-u1*u2)
            x1 = u1*x2
            q2 = q1 + (q4-q1)*x1
            q3 = q1 + (q4-q1)*x2
            kwargs = {
                "q1": q1,
                "q2": q2,
                "q3": q3,
                "q4": q4,
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
