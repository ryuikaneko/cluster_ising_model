import numpy as np

dat = np.load('coupling_IsingCluster_perturbed.npy')
np.savetxt("_coupling_IsingCluster_perturbed",dat)

gxs = np.linspace(0,4,25)
gzzs = np.linspace(0,4,25)
#print(gxs)
#print(gzzs)
dat2 = []
for gzz in gzzs:
    for gx in gxs:
        if gx+gzz < 4.0+1e-10:
            dat2.append([gx,gzz])
dat2 = np.array(dat2)
#print(dat2)
print(dat2.shape)
np.savetxt("_coupling_IsingCluster_perturbed_2",dat2)
