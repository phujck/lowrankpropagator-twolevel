import numpy as np
N = 12 # number of spins
# array of spin energy splittings and coupling strengths. here we use
# uniform parameters, but in general we don't have to
h_param=0.1*2*np.pi
Jz_param=0.1*2*np.pi
Jx_param=0.1*2*np.pi
Jy_param=0.1*2*np.pi
# init='mixed'
init='xbasis'
# dephasing rate
# gamma = 0.01 * np.ones(N)
dephase = 0
# bath coupling
bath_couple = 1e-4
driving = 1
steps = 3000
endtime=100
tlist, deltat = np.linspace(0, endtime, steps, retstep=True)

rank=1
rank2=4
ntraj = 5001
run_lowrank=True
run_exact=True
run_mc=True
# run_exact=False
# run_mc=False