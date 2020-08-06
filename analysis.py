from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import time
from params import *
import os
params = {
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [2 * 3.375, 2 * 3.375],
    'text.usetex': True,
    'figure.figsize': (18, 12),
    'lines.linewidth' : 2.5

}

plt.rcParams.update(params)


outfile_exact = './Data/exact:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu.npz'.format(
        N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving)
outfile_mc = './Data/montecarlo:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu-{}ntraj.npz'.format(
        N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving, ntraj)
outfile_lowrank = './Data/lowrank:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu-{}rank.npz'.format(
        N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving, rank)

expectations_exact= np.load(outfile_exact)
expectations_mc= np.load(outfile_mc)
expectations_lowrank= np.load(outfile_lowrank)

s_exact=expectations_exact['expectations']
# s_exact=expectations_mc['expectations']
s_mc=expectations_mc['expectations']
s_lowrank=expectations_lowrank['expectations']
print(expectations_mc['runtime'])
print(expectations_lowrank['runtime'])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
for n in range(N):
    ax1.plot(tlist, np.real(s_exact[n]), label='$\\langle\\sigma_z^{(%d)}\\rangle$' % n)
    ax1.plot(tlist, np.real(s_lowrank[:,n]), linestyle='--',label='Low-rank $\\langle\\sigma_z^{(%d)}\\rangle$' % n)
    ax2.plot(tlist, np.real(s_lowrank[:,N+n]), linestyle='--',label='Low-rank $\\langle\\sigma_z^{(%d)}\\rangle$' % n)
    ax2.plot(tlist, np.real(s_exact[N + n]), label='$\\langle\\sigma_x^{(%d)}\\rangle$' % n)
    # ax2.plot(tlist, np.real(s_rank[:,N + n]),linestyle='--', label='Low-rank $\\langle\\sigma_x^{(%d)}\\rangle$' % n)
# ax1.legend(loc=0)
ax2.set_xlabel('Time')
ax1.set_ylabel('$\\langle\sigma_z\\rangle$')
ax2.set_ylabel('$\\langle\sigma_z\\rangle$')
ax2.legend()
ax1.legend()

plt.show()

plt.subplot(211)
for n in range(N-1):
    plt.plot(tlist, np.real(s_exact[n]), color='red')
    plt.plot(tlist, np.real(s_lowrank[:,n]), color='blue', linestyle='--')
plt.plot(tlist, np.real(s_exact[N-1]), color='red', label='$\\langle\\sigma_z^{(j)}\\rangle$' )
plt.plot(tlist, np.real(s_lowrank[:, N-1]), color='blue',   linestyle = '--', label = 'Low-rank $\\langle\\sigma_z^{(j)}\\rangle$' )
# plt.xlabel('Time')
plt.ylabel('$\\langle\sigma_z\\rangle$')
plt.legend()
plt.grid(True)
# plt.xlim(0,95)

plt.subplot(212)
for n in range(N-1):
    plt.plot(tlist, np.real(s_exact[n]), color='red')
    plt.plot(tlist, np.real(s_mc[n]), color='blue', linestyle='--')
plt.plot(tlist, np.real(s_exact[N-1]), color='red', label='$\\langle\\sigma_z^{(j)}\\rangle$' )
plt.plot(tlist, np.real(s_mc[N-1]), color='blue',   linestyle = '--', label = 'Monte-Carlo $\\langle\\sigma_z^{(j)}\\rangle$' )
plt.legend()
plt.xlabel('Time')
plt.grid(True)
plt.show()