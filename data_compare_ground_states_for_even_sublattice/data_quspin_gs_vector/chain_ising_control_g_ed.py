from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np # general math functions
import scipy.sparse.linalg
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="cluster ising")
    parser.add_argument("-L",metavar="L",dest="L",type=int,default=10,help="set L")
#    parser.add_argument("-gzxz",metavar="gzxz",dest="gzxz",type=np.float64,default=1.0,help="set gzxz")
#    parser.add_argument("-gzz",metavar="gzz",dest="gzz",type=np.float64,default=1.0,help="set gzz")
#    parser.add_argument("-gx",metavar="gx",dest="gx",type=np.float64,default=1.0,help="set gx")
    parser.add_argument("-gzxz",metavar="gzxz",dest="gzxz",type=np.float64,default=3.0625,help="set gzxz")
    parser.add_argument("-gzz",metavar="gzz",dest="gzz",type=np.float64,default=0.875,help="set gzz")
    parser.add_argument("-gx",metavar="gx",dest="gx",type=np.float64,default=0.0625,help="set gx")
    return parser.parse_args()

def calc_gs(gzxz,gzz,gx,L):
#    basis = spin_basis_1d(L=L,Nup==L//2,a=1,pauli=1,kblock=0,pblock=1)
    basis = spin_basis_1d(L=L,a=1,pauli=1)
    gzxzs =[[+gzxz,(i-1)%L,i,(i+1)%L] for i in range(L)]
    gzzs =[[-gzz,i,(i+1)%L] for i in range(L)]
    gxs = [[-gx,i] for i in range(L)]
    static = [["zxz",gzxzs],["zz",gzzs],["x",gxs]]
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    H = hamiltonian(static,[],static_fmt="csr",basis=basis,dtype=np.float64,**no_checks)
#    ene, vec = H.eigsh(time=0.0,which="SA",k=5)
    ene, vec = H.eigsh(time=0.0,which="SA",k=2)
    vec = vec[:,0]
    return ene, vec

def main():
    args = parse_args()
    L = args.L
    gzxz = args.gzxz
    gzz = args.gzz
    gx = args.gx
    dat = []
#    gs = np.linspace(-1,1,33)
#    for g in gs:
#        gzxz = (g-1)**2
#        gzz = 2*(1-g**2)
#        gx = (1+g)**2
#        ene, vec = calc_gs(gzxz,gzz,gx,L)
#        print(g,*ene/L)
#        dat.append([g,*ene/L])
#        #print(vec)
    ene, vec = calc_gs(gzxz,gzz,gx,L)
    print(gzz,gx,gzxz,*ene/L)
    dat.append([gzz,gx,gzxz,*ene/L])
    vec *= np.sign(vec[0].real) # set a first component of a vector positive
#    np.savetxt("dat_L"+"{}".format(L),dat)
    np.savetxt("dat_ene_L"+"{}".format(L),dat)
    np.savetxt("dat_vec_L"+"{}".format(L),vec)

    vecLrank = vec.reshape(L*[2])
#    vec2 = vecLrank.reshape(1,-1)[0]
#    print(vec)
#    print(vecLrank)
#    print(vec2)
#    print(np.array_equal(vec,vec2))
    np.savez("dat_vecLrank_L"+"{}".format(L),vecLrank)

if __name__ == "__main__":
    main()
