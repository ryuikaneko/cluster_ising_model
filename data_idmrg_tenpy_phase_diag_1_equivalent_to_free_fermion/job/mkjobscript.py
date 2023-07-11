import numpy as np

gxs = np.linspace(0,4,25)
gzzs = np.linspace(0,4,25)
#print(gxs)
#print(gzzs)
datpair = []
cnt = 0
for gzz in gzzs:
    for gx in gxs:
        if gx+gzz < 4.0+1e-10:
            datpair.append([gx,gzz])
            gzxz = 4.0 - gx - gzz
            f = open("_job_"+"{}".format(str(cnt).zfill(4))+".sh","w")
            f.write("#!/usr/bin/bash\n")
            f.write("#BSUB -J test\n")
            f.write("#BSUB -e test.%J.err\n")
            f.write("#BSUB -o test.%J.out\n")
            f.write("#BSUB -q normal\n")
            f.write("#BSUB -n 1\n")
            f.write("#BSUB -m laguerre03\n")
            f.write("env MKL_NUM_THREADS=1\n")
            f.write("source ${HOME}/miniconda3/etc/profile.d/conda.sh\n")
            f.write("conda activate tenpy\n")
            f.write("date\n")
            f.write("python ../chain_ising_control_g_idmrg.py -gzz "+"{:.10f}".format(gzz)+" -gx "+"{:.10f}".format(gx)+" -gzxz "+"{:.10f}".format(gzxz)+"\n")
            f.write("date\n")
            f.write("conda deactivate\n")
            f.close()
            cnt += 1
datpair = np.array(datpair)
#print(datpair)
print(datpair.shape)
np.savetxt("datpair",datpair)
