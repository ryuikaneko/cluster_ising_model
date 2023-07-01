# https://tenpy.readthedocs.io/en/latest/intro/model.html
# https://tenpy.readthedocs.io/en/latest/examples/model_custom.html

# how to write a three-site interaction?
# https://tenpy.johannes-hauschild.de/viewtopic.php?t=75
## deprecated?

# About four-site coupling terms
# https://tenpy.johannes-hauschild.de/viewtopic.php?t=78
# https://github.com/tenpy/tenpy/blob/main/tenpy/models/toric_code.py#L91-L158

# add_multi_coupling_term
# https://tenpy.readthedocs.io/en/latest/reference/tenpy.models.model.MultiCouplingModel.html#tenpy.models.model.MultiCouplingModel.add_multi_coupling_term
# https://tenpy.readthedocs.io/en/latest/reference/tenpy.models.model.CouplingModel.html#tenpy.models.model.CouplingModel.add_multi_coupling_term

"""Prototypical example of a quantum model: the transverse field Ising model.

Like the :class:`~tenpy.models.xxz_chain.XXZChain`, the transverse field ising chain
:class:`TFIChain` is contained in the more general :class:`~tenpy.models.spins.SpinChain`;
the idea is more to serve as a pedagogical example for a 'model'.

We choose the field along z to allow to conserve the parity, if desired.
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

import numpy as np

#from .model import CouplingMPOModel, NearestNeighborModel
#from .lattice import Chain
#from ..tools.params import asConfig
#from ..networks.site import SpinHalfSite
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
from tenpy.tools.params import asConfig
from tenpy.networks.site import SpinHalfSite

from tenpy.networks.mps import MPS
#from tenpy.models.tf_ising import TFIChain
#from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg

import logging
logging.basicConfig(level=logging.INFO)

## https://tenpy.readthedocs.io/en/latest/intro/input_output.html
## example:
##
## import h5py
## from tenpy.tools import hdf5_io
## 
## data = {"psi": psi,  # e.g. an MPS
##         "model": my_model,
##         "parameters": {"L": 6, "g": 1.3}}
## 
## with h5py.File("file.h5", 'w') as f:
##     hdf5_io.save_to_hdf5(f, data)
## # ...
## with h5py.File("file.h5", 'r') as f:
##     data = hdf5_io.load_from_hdf5(f)
##     # or for partial reading:
##     pars = hdf5_io.load_from_hdf5(f, "/parameters")
## 
import h5py
from tenpy.tools import hdf5_io

import argparse


__all__ = ['CustomTFIModel', 'CustomTFIChain']


class CustomTFIModel(CouplingMPOModel):
    r"""Transverse field Ising model on a general lattice.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{gzz} \sigma^z_i \sigma^z_{j}
            - \sum_{i} \mathtt{gx} \sigma^x_i
            - \sum_{i} \mathtt{gzxz} \sigma^z_i \sigma^x_{i+1} \sigma^z_{i+2}

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`CustomTFIModel` below.

    Options
    -------
    .. cfg:config :: CustomTFIModel
        :include: CouplingMPOModel

        conserve : None | 'parity'
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        gzz, gx : float | array
            Coupling as defined for the Hamiltonian above.

    """
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', None)
        site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        gzz = np.asarray(model_params.get('gzz', 1.))
        gx = np.asarray(model_params.get('gx', 1.))
        gzxz = np.asarray(model_params.get('gzxz', 0.))
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-gzz, u1, 'Sigmaz', u2, 'Sigmaz', dx)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-gx, u, 'Sigmax')
        ## only for 1D
        for u in range(len(self.lat.unit_cell)):
            # self.add_multi_coupling(+gzxz, u, "Sigmaz", [(u, "Sigmax", 1), (u, "Sigmaz", 2)]) ## deprecated
            self.add_multi_coupling(+gzxz, [("Sigmaz", [0], u), ("Sigmax", [1], u), ("Sigmaz", [2], u)]) ## [("operator",dx,unitcell), ...]
        # done


#class CustomTFIChain(CustomTFIModel, NearestNeighborModel): ## not working for gzxz!=0 because it is no longer a n.n. model
class CustomTFIChain(CustomTFIModel):
    """The :class:`CustomTFIModel` on a Chain, suitable for TEBD.

    See the :class:`CustomTFIModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True


def example_DMRG_tf_ising_infinite(g):
    print("infinite DMRG, transverse field Ising model")
    print("g={g:.2f}".format(g=g))
    model_params = dict(L=2, gzz=2.0*(1.0-g**2), gx=(1.0+g)**2, gzxz=(g-1.0)**2, bc_MPS='infinite', conserve=None)
    M = CustomTFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
#    product_state = (["up", "down"] * (M.lat.N_sites))[:M.lat.N_sites] ## https://tenpy.johannes-hauschild.de/viewtopic.php?t=30
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-10,
    }
    # Sometimes, we want to call a 'DMRG engine' explicitly
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
#    print("E = {E:.13f}".format(E=E))
#    print("final bond dimensions: ", psi.chi)
#    mag_x = np.mean(psi.expectation_value("Sigmax"))
#    mag_z = np.mean(psi.expectation_value("Sigmaz"))
#    print("<sigma_x> = {mag_x:.5f}".format(mag_x=mag_x))
#    print("<sigma_z> = {mag_z:.5f}".format(mag_z=mag_z))
#    print("correlation length:", psi.correlation_length())
    return E, psi, M

#----

def parse_args():
    parser = argparse.ArgumentParser(description="cluster ising")
#    parser.add_argument("-L",metavar="L",dest="L",type=int,default=10,help="set L")
#    parser.add_argument("-gzxz",metavar="gzxz",dest="gzxz",type=np.float64,default=1.0,help="set gzxz")
#    parser.add_argument("-gzz",metavar="gzz",dest="gzz",type=np.float64,default=1.0,help="set gzz")
#    parser.add_argument("-gx",metavar="gx",dest="gx",type=np.float64,default=1.0,help="set gx")
    return parser.parse_args()

def main():
    args = parse_args()
#    L = args.L
#    gzxz = args.gzxz
#    gzz = args.gzz
#    gx = args.gx
    dat = []
#    gs = np.linspace(-1,1,5)
    gs = np.linspace(-1,1,33)
    for g in gs:
        E, psi, M = example_DMRG_tf_ising_infinite(g)
        print(g,E,max(psi.chi),psi.correlation_length())
        dat.append([g,E,max(psi.chi),psi.correlation_length()])
#
## print MPS
#        print(psi.get_B(0),psi.get_B(1))
#
## save MPS
#        datall = {
#            "psi": psi,
#            "model": M,
##            "parameters": {"L": L, "g": g}
#            "parameters": {"g": g}
#            }
#        with h5py.File("dat_g"+"{:.10f}".format(g)+".h5",'w') as f:
#            hdf5_io.save_to_hdf5(f,datall)
#
    np.savetxt("dat",dat,header="E,chi,corrlength")

if __name__ == "__main__":
    main()
