""" Latin-hypercube parameter design """

import logging
import math
from pathlib import Path
import re
import subprocess

import numpy as np

from . import cachedir, parse_system


def generate_lhs(npoints, ndim, seed):
    """
    Generate a maximin LHS.

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
    def __init__(self, system, npoints=500, seed=751783496):
        self.system = system

        self.projectiles, self.beam_energy = parse_system(system)

        # 5.02 TeV has ~1.2x particle production as 2.76 TeV
        # [https://inspirehep.net/record/1410589]
        norm_range = {
            2760: (8., 20.),
            5020: (10., 25.),
        }[self.beam_energy]

        # this label is so much longer than all the others
        # etas_slp_label = r'{atan}(\eta/s {slope}) [{GeV}^{-1}]'
        etas_slp_label = r'\eta/s {slope} [{GeV}^{-1}]'
        # upper bound of atan(slope)
        pi_2 = math.pi/2

        self.keys, labels, self.range = map(list, zip(*[
            ('norm',          r'{Norm}',                  (norm_range   )),
            ('trento_p',      r'p',                       ( -0.5,    0.5)),
            ('fluct_std',     r'\sigma {fluct}',          (  0.0,    2.0)),
            ('nucleon_width', r'w [{fm}]',                (  0.4,    1.0)),
            ('dmin3',         r'd_{min}^3 [{fm}^3]',      (  0.0, 1.7**3)),
            ('tau_fs',        r'\tau {fs} [{fm}/c]',      (  0.0,    1.5)),
            ('etas_hrg',      r'\eta/s {hrg}',            (  0.1,    0.5)),
            ('etas_min',      r'\eta/s {min}',            (  0.0,    0.2)),
            ('etas_slp_atan', etas_slp_label,             (  0.0,   pi_2)),
            ('etas_crv',      r'\eta/s {crv}',            ( -1.0,    1.0)),
            ('zetas_max',     r'\zeta/s {max}',           (  0.0,    0.1)),
            ('zetas_width',   r'\zeta/s {width} [{GeV}]', (  0.0,    0.1)),
            ('zetas_t0',      r'\zeta/s T_0 [{GeV}]',     (0.150,  0.200)),
            ('Tswitch',       r'T {switch} [{GeV}]',      (0.135,  0.165)),
        ]))

        # convert labels into TeX:
        #   - wrap normal text with \mathrm{}
        #   - escape spaces
        #   - surround with $$
        self.labels = [
            re.sub(r'({[A-Za-z]+})', r'\mathrm\1', i)
            .replace(' ', r'\ ')
            .join('$$')
            for i in labels
        ]

        self.min, self.max = map(np.array, zip(*self.range))

        # use padded numbers for design point names
        fmt = '{:0' + str(len(str(npoints - 1))) + 'd}'
        self.points = [fmt.format(i) for i in range(npoints)]

        self.array = self.min + (self.max - self.min)*generate_lhs(
            npoints=npoints, ndim=len(self.keys), seed=seed
        )

        # return

        i = self.keys.index('etas_slp_atan')
        # slope_max = 20

        self.array[:, i] = np.tan(self.array[:, i])
        slope_max = 8.
        self.range[i] = (0., slope_max)
        self.max[i] = slope_max
        remove = set((self.array[:, i] > 10).nonzero()[0])

        # atan_max = math.atan(slope_max)
        # remove = set((self.array[:, i] > atan_max).nonzero()[0])
        # self.range[i] = (0., atan_max)

        # self.array[:, i] = np.log10(1 + np.tan(self.array[:, i]))
        # self.range[i] = (0., math.log10(1 + slope_max))

        # log1p_max = math.log(1 + slope_max)
        # self.array[:, i] = np.log(1 + np.tan(self.array[:, i]))
        # self.range[i] = (0., log1p_max)
        # remove = set((self.array[:, i] > log1p_max).nonzero()[0])

        i = self.keys.index('tau_fs')
        remove.update(set((self.array[:, i] < .02).nonzero()[0]))
        # self.range[i] = (tau_min, self.range[i][1])

        # remove outliers
        # TODO fix and re-run?
        outliers = {281, 340}
        remove.update(outliers)
        self.points = [
            p for n, p in enumerate(self.points)
            if n not in remove
        ]
        self.array = np.delete(self.array, list(remove), axis=0)
        # self.array[i, self.keys.index('tau_fs')] = 1e-3
        # self.points[i] += '-2'

    def __array__(self):
        return self.array

    _template = ''.join(
        '{} = {}\n'.format(key, ' '.join(args)) for (key, *args) in
        [[
            'trento-args',
            '{projectiles[0]} {projectiles[1]}',
            '--cross-section {cross_section}',
            '--normalization {norm}',
            '--reduced-thickness {trento_p}',
            '--fluctuation {fluct}',
            '--nucleon-min-dist {dmin}',
        ], [
            'nucleon-width', '{nucleon_width}'
        ], [
            'tau-fs', '{tau_fs}'
        ], [
            'hydro-args',
            'etas_hrg={etas_hrg}',
            'etas_min={etas_min}',
            'etas_slope={etas_slp}',
            'etas_curv={etas_crv}',
            'zetas_max={zetas_max}',
            'zetas_width={zetas_width}',
            'zetas_t0={zetas_t0}',
        ], [
            'Tswitch', '{Tswitch}'
        ]]
    )

    def write_files(self, basedir):
        outdir = basedir / self.system
        outdir.mkdir(parents=True, exist_ok=True)

        for point, row in zip(self.points, self.array):
            kwargs = dict(
                zip(self.keys, row),
                projectiles=self.projectiles,
                cross_section={
                    # sqrt(s) [GeV] : sigma_NN [fm^2]
                    200: 4.2,
                    2760: 6.4,
                    5020: 7.0,
                }[self.beam_energy]
            )
            kwargs.update(
                fluct=1/kwargs.pop('fluct_std')**2,
                dmin=kwargs.pop('dmin3')**(1/3),
                etas_slp=math.tan(kwargs.pop('etas_slp_atan')),
            )
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

    for s in systems:
        Design(s).write_files(args.inputs_dir)

    logging.info('wrote all files to %s', args.inputs_dir)


if __name__ == '__main__':
    main()
