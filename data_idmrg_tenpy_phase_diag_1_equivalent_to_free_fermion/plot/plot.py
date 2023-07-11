import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt("dat")

datx = dat[:,1] + 0.5*dat[:,0]
daty = 0.5*np.sqrt(3.0)*dat[:,0]
datr = np.abs(dat[:,6])
datr = datr/np.max(datr)
datg = np.abs(dat[:,7])
datg = datg/np.max(datg)
datb = np.abs(dat[:,8])
datb = datb/np.max(datb)
datalpha = 1-np.min([datr,datg,datb],axis=0)
datalpha = (datalpha-np.min(datalpha))/(np.max(datalpha)-np.min(datalpha))
#datc = np.array([datr,datg,datb,datalpha]).T ## with tranparancy
#datc = np.array([datr,datg,datb]).T ## without transparancy
datc = np.array([datr,datg*2.0/3.0,datb]).T ## without transparancy, darker green
#print(datc)
#print(np.min(datc[:,3]))
#print(np.max(datc[:,3]))

plt.figure(figsize=(5,5))
plt.scatter(datx,daty,s=48,c=datc,marker="o",lw=1,ec="k",zorder=20)
#plt.plot(datx,daty,'o',ms=6,mec='k',mew=1)

plt.plot([0,4],[0,0],"-",c="black",zorder=10)
plt.plot([4,2],[0,2.0*np.sqrt(3.0)],"-",c="black",zorder=10)
plt.plot([2,0],[2.0*np.sqrt(3.0),0],"-",c="black",zorder=10)
plt.text(0,-0.25,r"$0$",ha="center",va="center")
plt.text(1,-0.25,r"$1$",ha="center",va="center")
plt.text(2,-0.25,r"$2$",ha="center",va="center")
plt.text(3,-0.25,r"$3$",ha="center",va="center")
plt.text(4,-0.25,r"$4$",ha="center",va="center")
plt.text(4.0+0.25,np.sqrt(3.0)*0.0,r"$0$",ha="center",va="center")
plt.text(3.5+0.25,np.sqrt(3.0)*0.5,r"$1$",ha="center",va="center")
plt.text(3.0+0.25,np.sqrt(3.0)*1.0,r"$2$",ha="center",va="center")
plt.text(2.5+0.25,np.sqrt(3.0)*1.5,r"$3$",ha="center",va="center")
plt.text(2.0+0.25,np.sqrt(3.0)*2.0,r"$4$",ha="center",va="center")
plt.text(2.0-0.25,np.sqrt(3.0)*2.0,r"$0$",ha="center",va="center")
plt.text(1.5-0.25,np.sqrt(3.0)*1.5,r"$1$",ha="center",va="center")
plt.text(1.0-0.25,np.sqrt(3.0)*1.0,r"$2$",ha="center",va="center")
plt.text(0.5-0.25,np.sqrt(3.0)*0.5,r"$3$",ha="center",va="center")
plt.text(0.0-0.25,np.sqrt(3.0)*0.0,r"$4$",ha="center",va="center")

plt.text(-0.375,0.625,"SPT",fontsize=16,c="blue",ha="center",va="center")
plt.text(4.375,0.625,"Trivial",fontsize=16,c="red",ha="center",va="center")
plt.text(2,4,"SB",fontsize=16,c="darkgreen",ha="center",va="center")
plt.text(-0.375,0.375,r"$|\langle Z_{i-2}Y_{i-1}X_{i}Y_{i+1}Z_{i+2}\rangle|$",fontsize=5,c="blue",ha="center",va="center")
plt.text(4.375,0.375,r"$|\langle X_i\rangle|$",fontsize=10,c="red",ha="center",va="center")
plt.text(2,3.75,r"$|\langle Z_i\rangle|$",fontsize=10,c="darkgreen",ha="center",va="center")
plt.text(2,-0.5,r"$g_x$",fontsize=16,ha="center",va="center")
plt.text(3.5,2.125,r"$g_{zz}$",fontsize=16,ha="center",va="center")
plt.text(0.5,2.125,r"$g_{zxz}$",fontsize=16,ha="center",va="center")

plt.xlim([-0.5,4.5])
plt.ylim([-0.5,4.5])
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')

plt.savefig("fig_phase_diag.pdf",bbox_inches="tight")
plt.close()
