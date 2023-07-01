


import numpy as np
import matplotlib.pyplot as plt


''' 
Plot the triangular phase diagram.

The "coupling" array stores the parameter array (g_x, g_zz).
The parameter g_zxz is related to them by: 
                g_zxz = 4 - g_x - g_zz

The "pred" array stores the prediction probability for each phase:

        (SB, Trivial, SPT, Unsuccessful)
when the QCNN is applied to the corresponding couplings.

'''



#path = None
path = './'
coupling = np.load(path + 'coupling_IsingCluster_perturbed.npy')
pred = np.load(path + 'pred_IsingCluster_perturbed.npy')

x = []
zz = []
for pair in coupling:
    x.append(pair[0]+pair[1]/2)
    zz.append(pair[1]*3**0.5/2)
    
x = np.array(x)
zz = np.array(zz)
gx0 = x[pred==0]
gx1 = x[pred==1]
gx2 = x[pred==2]

gzz0 = zz[pred==0]
gzz1 = zz[pred==1]
gzz2 = zz[pred==2]

plt.figure(figsize = (5,5))
plt.plot(gx0, gzz0,'o', markersize = 5, markeredgecolor = 'k', markerfacecolor = 'C0', markeredgewidth = 0)
plt.plot(gx1, gzz1,'o', markersize = 5, markeredgecolor = 'k', markerfacecolor = 'C1', markeredgewidth = 0)
plt.plot(gx2, gzz2,'o', markersize = 5, markeredgecolor = 'k', markerfacecolor = 'C2', markeredgewidth = 0)

plt.xlim([-0.5,4.5])
plt.ylim([-0.5,4.5])
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')


plt.savefig("fig_1.pdf",bbox_inches="tight")
plt.close()


''' 
Plot the prediction from H1 to H4 in the main text.

The "coupling" array stores the parameter array lambda.

The "pred" array stores the prediction probability for each phase:

        (SB, Trivial, SPT, Unsuccessful)
when the QCNN is applied to the corresponding coupling.

'''


#path = 'c:/users/user/desktop/QCNN_data/'
path = './'
coupling = np.load(path + 'coupling_H1.npy')
pred = np.load(path + 'pred_H1.npy')

plt.plot(coupling, pred[:,0],'C0-', label = 'GM')
plt.plot(coupling, pred[:,1],'C1-', label = 'GM')
plt.plot(coupling, pred[:,2],'C2-', label = 'GM')

plt.xlabel('$\lambda$')
plt.ylabel('Probability')


plt.savefig("fig_2.pdf",bbox_inches="tight")
plt.close()



























