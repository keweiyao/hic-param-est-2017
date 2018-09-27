"""
Downloads, processes, and stores experimental data.
Prints all data when run as a script.
"""

from collections import defaultdict
import logging
import pickle
import warnings
from urllib.request import urlopen

import numpy as np
import yaml, re

from . import cachedir, prelimdir, systems

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class HEPData:
    """
    Interface to a `HEPData <https://hepdata.net>`_ YAML data table.

    Downloads and caches the dataset specified by the INSPIRE record and table
    number.  The web UI for `inspire_rec` may be found at
    :file:`https://hepdata.net/record/ins{inspire_rec}`.

    If `reverse` is true, reverse the order of the data table (useful for
    tables that are given as a function of Npart).

    .. note::

        Datasets are assumed to be a function of centrality.  Other kinds of
        datasets will require code modifications.

    """
    def __init__(self, inspire_rec, table, reverse=False):
        cachefile = (
            cachedir / 'hepdata' /
            'ins{}_table{}.pkl'.format(inspire_rec, table)
        )
        name = 'record {} table {}'.format(inspire_rec, table)

        if cachefile.exists():
            logging.debug('loading from hepdata cache: %s', name)
            with cachefile.open('rb') as f:
                self._data = pickle.load(f)
        else:
            logging.debug('downloading from hepdata.net: %s', name)
            cachefile.parent.mkdir(exist_ok=True)
            with cachefile.open('wb') as f, urlopen(
                    'https://hepdata.net/download/table/'
                    'ins{}/Table{}/yaml'.format(inspire_rec, table)
            ) as u:
                self._data = yaml.load(u)
                pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if reverse:
            for v in self._data.values():
                for d in v:
                    d['values'].reverse()

    def x(self, name, case=True):
        """
        Get an independent variable ("x" data) with the given name.

        If `case` is false, perform case-insensitive matching for the name.

        """
        trans = (lambda x: x) if case else (lambda x: x.casefold())
        name = trans(name)

        for x in self._data['independent_variables']:
            if trans(x['header']['name']) == name:
                return x['values']


    @property
    def pT(self):
        """
        The pT bins as a list of (low, high) tuples.

        """
        try:
            return self._pT
        except AttributeError:
            pass

        for name in ['PT (GeV)', 'PT', 'electron $\\it{p}_{T} (GeV/\\it{c})$',
                      '$p_{\\rm T}$', '<pT> +-(dx)']:
            x = self.x(name, case=False)

            if x != None:
               break
        if x is None:
            raise LookupError(bcolors.FAIL+"no x data with name '{}'".format(name)+bcolors.ENDC)

        try:
            pT = [(v['low'], v['high']) for v in x]
        except KeyError:
            # try to guess bins from midpoints
            if np.isreal(x[0]['value']):
                mids = [v['value'] for v in x]
                width = set(a - b for a, b in zip(mids[1:], mids[:-1]))
                if len(width) > 1:
                    d = 1
                else:
                    d = width.pop() / 2
                pT = [(m - d, m + d) for m in mids]
            else:
                ll = [re.split(' |\+|\,-', v['value']) for v in x]
                pT = [(float(l[0])-float(l[2]), float(l[0])+float(l[2]))
                       for l in ll]

        self._pT = pT

        return pT

    @pT.setter
    def pTcent(self, value):
        """
        Manually set centrality bins.

        """
        self._pT = value

    def y(self, name=None, **quals):
        """
        Get a dependent variable ("y" data) with the given name and qualifiers.

        """
        for y in self._data['dependent_variables']:
            if name is None or y['header']['name'].startswith(name):
                y_quals = {q['name']: q['value'] for q in y['qualifiers']}
                if all(y_quals[k] == v for k, v in quals.items()):
                    return y['values']

        raise LookupError(
            bcolors.FAIL+"no y data with name '{}' and qualifiers '{}'"
            .format(name, quals)+bcolors.ENDC
        )

    def dataset(self, name=None, ignore_bins=[], **quals):
        """
        Return a dict containing:

        - **cent:** list of centrality bins
        - **x:** numpy array of centrality bin midpoints
        - **y:** numpy array of y values
        - **yerr:** subdict of numpy arrays of y errors

        `name` and `quals` are passed to `HEPData.y()`.

        Missing y values are skipped.

        Centrality bins whose upper edge is greater than `maxcent` are skipped.

        Centrality bins in `ignore_bins` [a list of (low, high) tuples] are
        skipped.

        """
        pT = []
        y = []
        yerr = defaultdict(list)

        for ipT, v in zip(self.pT, self.y(name, **quals)):
            # skip missing values
            # skip bins whose upper edge is greater than maxcent
            # skip explicitly ignored bins
            if v['value'] == '-' or ipT in ignore_bins:
                continue

            pT.append(ipT)
            y.append(v['value'])

            for err in v['errors']:
                try:
                    e = err['symerror']
                except KeyError:
                    e = err['asymerror']
                    if abs(e['plus']) != abs(e['minus']):
                        warnings.warn(bcolors.WARNING + 'Asymmetric errors are not implemented'+ bcolors.ENDC)
                    continue


                yerr[err.get('label', 'sum')].append(e)
        pT = np.array(pT)
        x=np.array([(a + b)/2 for (a, b) in pT])
        return dict(
            pT=pT,
            x=x,
            y=np.array(y),
            yerr={k: np.array(v) for k, v in yerr.items()},
        )

def cut(d, pTcut=10.):
    for k in sorted(d):
        v = d[k]
        for a in v:
            for b in v[a]:
                for c in v[a][b]:
                    iv = v[a][b][c]
                    cut = iv['x']>pTcut
                    iv['x'] = iv['x'][cut]
                    iv['y'] = iv['y'][cut]
                    iv['pT'] = iv['pT'][cut]
                    iv['yerr']['stat'] = iv['yerr']['stat'][cut]
                    iv['yerr']['sys'] = iv['yerr']['sys'][cut]

def _data():
    """
    Curate the experimental data using the `HEPData` class and return a nested
    dict with levels

    - system
    - observable
    - subobservable
    - dataset (created by :meth:`HEPData.dataset`)

    For example, ``data['PbPb2760']['dN_dy']['pion']`` retrieves the dataset
    for pion dN/dy in Pb+Pb collisions at 2.76 TeV.

    Some observables, such as charged-particle multiplicity, don't have a
    natural subobservable, in which case the subobservable is set to `None`.

    The best way to understand the nested dict structure is to explore the
    object in an interactive Python session.

    """
    data = {s: {} for s in systems}

    # Naming scheme for heavy quark observable
    # System/Observable/Particle-species/Centrality->dataset as function of pT
    # ALICE, p+p, sqrts=7TeV, D meson spectra

    ####### SQRTS = 7000 GeV #####################
    if 'pp7000' in systems:
        data['pp7000'].update({'dX/dp/dy': {}})
        # 1) Particle spectra
        for i, D in enumerate(['D0', 'D+', 'D*'], start=1):
            dset = HEPData(1511870, i+1).dataset('d$\\sigma$/d $p_{\\rm{T}}$dy')
            data['pp7000']['dX/dp/dy'][D] = {'MB': dset}

    ####### SQRTS = 2760 GeV #####################
    if 'PbPb2760' in systems:
        data['PbPb2760'].update({'V2': {'D-avg':{}, 'HF->e+e-':{}, 'D0':{}},
                                 'RAA': {'HF->e+e-':{}, 'D-avg':{}, 'B-avg':{}}
                                       })

        # 1) ALICE, Pb+Pb, Flow, D meson
        dset = HEPData(1233087, 4).dataset('V2')
        data['PbPb2760']['V2']['D-avg'].update({'30-50': dset})

        # 2) ALICE, Pb+Pb, Flow, HF -> e+e-
        for i, cen in enumerate(['0-10', '10-20', '20-40'], start=1):
            dset = HEPData(1466626, i).dataset("v2 +-(stat) +(systUncorr) - (systUncorr)")
            data['PbPb2760']['V2']['HF->e+e-'].update({cen: dset})

        # 3) ALICE, Pb+Pb, Flow, D0
        for i, cen in enumerate(['0-10', '10-20', '30-50'], start=1):
            dset = HEPData(1294938, i).dataset("V2")
            data['PbPb2760']['V2']['D0'].update({cen: dset})

        # 4) ALICE, Pb+Pb, RAA, D meson
        for i, cen in enumerate(['0-10', '30-50'], start=15):
            dset = HEPData(1394580, i).dataset('$R_{\\rm AA}$')
            data['PbPb2760']['RAA']['D-avg'].update({cen: dset})

        # 5) ALICE, Pb+Pb, RAA, c, b hadron to e+e-
        for i, cen in enumerate(['0-10', '10-20', '20-30',
                       '30-40', '40-50', '50-80'], start=7):
            dset = HEPData(1487727, i).dataset('$R_{AA}$')
            data['PbPb2760']['RAA']['HF->e+e-'].update({cen: dset})


    ####### SQRTS = 5020 GeV #####################
    if 'PbPb5020' in systems:
        data['PbPb5020']['ALICE'] =  {'V2': {'D-avg': {}},
                                     'RAA': {'D-avg': {}}
                                    }
        data['PbPb5020']['CMS'] =  {'V2': {'D0': {}},
                                     'RAA': {'D0': {}, 'B':{}}
                                    }
        # 1) ALICE, Pb+Pb, meson flow
        dset = HEPData(1608612, 5).dataset('$v_2$')
        data['PbPb5020']['ALICE']['V2']['D-avg'].update({'30-50': dset})

        # 2) Prelim Data! ALICE, Pb+Pb, D Raa
        for cen in ['0-10','30-50','60-80']:
            pTL, pTH, raa, stat, sys = np.loadtxt(prelimdir/'ALICE-Raa-D-{}.dat'.format(cen)).T
            dset = {'pT':np.array([(pl, ph) for pl, ph in zip(pTL, pTH)]),
                    'x' : (pTL+pTH)/2.,
                    'y' : raa,
                    'yerr': { 'stat': stat,
                              'sys': sys}
                    }
            data['PbPb5020']['ALICE']['RAA']['D-avg'].update({cen: dset})

        for cen in ['30-50-L','30-50-H']:
            pTL, pTH, raa, stat, sys = np.loadtxt(prelimdir/'ALICE-V2-D-{}.dat'.format(cen)).T
            dset = {'pT':np.array([(pl, ph) for pl, ph in zip(pTL, pTH)]),
                    'x' : (pTL+pTH)/2.,
                    'y' : raa,
                    'yerr': { 'stat': stat,
                              'sys': sys}
                    }
            data['PbPb5020']['ALICE']['V2']['D-avg'].update({cen: dset})


        # 3) CMS, Pb+Pb, D0 flow
        for cen in ['0-10','10-30','30-50']:
            pTL, pTH, v2, stat, sys1, sys2 = np.loadtxt('./official-exp/CMS-v2-D-{}.dat'.format(cen)).T
            dset = {'pT':np.array([(pl, ph) for pl, ph in zip(pTL, pTH)]),
                    'x' : (pTL+pTH)/2.,
                    'y' : v2,
                    'yerr': { 'stat': stat,
                              'sys': sys1,
                              }#'sys2': sys2}
                    }
            data['PbPb5020']['CMS']['V2']['D0'].update({cen: dset})

        # 4) CMS, Pb+Pb, D0 RAA
        for cen in ['0-10','0-100']:
            pTL, pTH, RAA, stat, syserror = np.loadtxt('./official-exp/CMS-Raa-D-{}.dat'.format(cen)).T
            dset = {'pT':np.array([(pl, ph) for pl, ph in zip(pTL, pTH)]),
                    'x' : (pTL+pTH)/2.,
                    'y' : RAA,
                    'yerr': { 'stat': stat,
                              'sys': syserror}
                    }
            data['PbPb5020']['CMS']['RAA']['D0'].update({cen: dset})
        # 5) CMS, Pb+Pb, B+/- RAA
        for cen in ['0-100']:
            pTL, pTH, RAA, stat, syserror = np.loadtxt('./official-exp/CMS-Raa-B-{}.dat'.format(cen)).T
            dset = {'pT':np.array([(pl, ph) for pl, ph in zip(pTL, pTH)]),
                    'x' : (pTL+pTH)/2.,
                    'y' : RAA,
                    'yerr': { 'stat': stat,
                              'sys': syserror}
                    }
            data['PbPb5020']['CMS']['RAA']['B'].update({cen: dset})

    #cut(data['PbPb5020'], 10.0)
    return data


def _baseline_data():
    """
    Curate the experimental data using the `HEPData` class and return a nested
    dict with levels

    - system
    - observable
    - subobservable
    - dataset (created by :meth:`HEPData.dataset`)

    For example, ``data['PbPb2760']['dN_dy']['pion']`` retrieves the dataset
    for pion dN/dy in Pb+Pb collisions at 2.76 TeV.

    Some observables, such as charged-particle multiplicity, don't have a
    natural subobservable, in which case the subobservable is set to `None`.

    The best way to understand the nested dict structure is to explore the
    object in an interactive Python session.

    """
    data = {'pp7000': {}, 'pp5020': {} }


    ####### SQRTS = 7000 GeV #####################
    data['pp7000'].update({'dX/dp/dy': {}})
    # 1) Particle spectra
    for i, D in enumerate(['D0', 'D+', 'D*'], start=1):
        dset = HEPData(1511870, i).dataset('d$\\sigma$/d $p_{\\rm{T}}$dy')
        data['pp7000']['dX/dp/dy'][D] = {'MB': dset}

    data['pp5020'].update({'dX/dp/dy': {}})
    # 1) Particle spectra
    for i, B in enumerate(['B+'], start=1):
        dset = HEPData(1599548, i).dataset('$\\frac{d \\sigma}{dp_{T}}$ (Pb/(GeV/C))')
        dset['pT'] = [(7.,10.),(10.,15.),(15.,20.),(20.,30.),(30.,50.)]
        data['pp5020']['dX/dp/dy'][B] = {'MB': dset}

    return data

#: A nested dict containing all the experimental data, created by the
#: :func:`_data` function.
data = _data()
ppdata = _baseline_data()


def cov(
        system1, exp1, obs1, specie1, cen1,
        system2, exp2, obs2, specie2, cen2,
        stat_frac=1e-4, log_pT_corr_length=.01, cross_factor=0.6,
):
    """
    Estimate a covariance matrix for the given system and pair of observables,
    e.g.:

    >>> cov('PbPb2760', 'dN_dy', 'pion', 'dN_dy', 'pion')
    >>> cov('PbPb5020', 'dN_dy', 'pion', 'dNch_deta', None)

    For each dataset, stat and sys errors are used if available.  If only
    "summed" error is available, it is treated as sys error, and `stat_frac`
    sets the fractional stat error.

    Systematic errors are assumed to have a Gaussian correlation as a function
    of centrality percentage, with correlation length set by `sys_corr_length`.

    If obs{1,2} are the same but subobs{1,2} are different, the sys error
    correlation is reduced by `cross_factor`.

    If obs{1,2} are different and uncorrelated, the covariance is zero.  If
    they are correlated, the sys error correlation is reduced by
    `cross_factor`.  Two different obs are considered correlated if they are
    both a member of one of the groups in `corr_obs` (the groups must be
    set-like objects).  By default {Nch, ET, dN/dy} are considered correlated
    since they are all related to particle / energy production.

    """

    def unpack(system, exp, obs, subobs, cen):
        dset = data[system][exp][obs][subobs][cen]
        yerr = dset['yerr']

        try:
            stat = yerr['stat']
            sys = yerr['sys']
        except KeyError:
            stat = dset['y'] * stat_frac
            sys = yerr['sum']

        return dset['x'], dset['y'], stat, sys

    x1, y1, stat1, sys1 = unpack(system1, exp1, obs1, specie1, cen1)
    x2, y2, stat2, sys2 = unpack(system2, exp2, obs2, specie2, cen2)

    same_obs = (obs1 == obs2) and (system1==system2) and (specie1 == specie2) \
                    and (cen1 == cen2) and (exp1==exp2)
    same_class = (obs1 == obs2) and (system1==system2) and (exp1==exp2)
    if (not same_obs) and (not same_class):
        return np.zeros([x1.size, x2.size])

    # compute the sys error covariance
    C = (
        np.exp(-.5*(np.subtract.outer(np.log(x1), np.log(x2))/log_pT_corr_length)**2) *
        np.outer(sys1, sys2)
    )

    if (not same_obs) and same_class:
        C *= cross_factor

    if same_obs:
        # add stat error to diagonal
        C.flat[::C.shape[0]+1] += stat1**2
        
    return C


def print_data(d, indent=0):
    """
    Pretty print the nested data dict.

    """
    prefix = indent * '  '
    for k in sorted(d):
        v = d[k]
        k = prefix + str(k)
        if isinstance(v, dict):
            print(k)
            print_data(v, indent + 1)
        else:
            if k.endswith('pT'):
                v = ' '.join(
                    str(tuple(int(j) if j.is_integer() else j for j in i))
                    for i in v
                )
            elif isinstance(v, np.ndarray):
                v = str(v).replace('\n', '')
            print(k, '=', v)

def plot_data(d, indent=0):
    """
    Ugly plot of the data
    """
    import matplotlib.pyplot as plt
    for k in sorted(d):
        v = d[k]
        for a in v:
            print(a)
            for b in v[a]:
                print(b)
                for c in v[a][b]:
                    print(c)
                    iv = v[a][b][c]
                    plt.errorbar(iv['x'], iv['y'], yerr=iv['yerr']['stat'],
                                 label="{}/{}/{}/{}".format(k,a,b,c))

                plt.legend(framealpha=0)
                plt.show()


def test_cov():
    covm = cov(
        'PbPb5020', 'CMS', 'RAA', 'D0', '0-10', 'PbPb5020', 'CMS', 'RAA', 'B', '0-100', )
    import matplotlib.pyplot as plt
    plt.imshow(np.flipud(covm.T))
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    print_data(ppdata)
    test_cov()
    import pickle
    with open('exp.pkl','bw') as f:
        pickle.dump(data, f)
    plot_data(data['PbPb5020'])
    
