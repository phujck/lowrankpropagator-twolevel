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
    'lines.linewidth' : 3,
    'lines.markersize' : 15

}

plt.rcParams.update(params)
# for dephase in [1e-4,1e-3,0]:
fig_params='{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu-{}rank-{}ntraj.pdf'.format(
        N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving, rank,ntraj)
outfile_exact = './Data/exact:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu.npz'.format(
        N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving)
outfile_mc = './Data/montecarlo:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu-{}ntraj.npz'.format(
        N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving, ntraj)
outfile_lowrank = './Data/lowrank:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu-{}rank.npz'.format(
        N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving, rank)
outfile_lowrank2 = './Data/lowrank:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu-{}rank.npz'.format(
        N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving, rank2)
# outfile_lowrank = './Data/lowrank:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu-{}rank.npz'.format(
#         N, init, h_param, Jx_param, Jy_param, mJz_param, endtime, steps, dephase, 1e-4, driving, rank)
expectations_exact= np.load(outfile_exact)
expectations_mc= np.load(outfile_mc)
expectations_lowrank= np.load(outfile_lowrank)
expectations_lowrank2= np.load(outfile_lowrank2)

s_exact=expectations_exact['expectations']
# s_exact=expectations_mc['expectations']
s_mc=expectations_mc['expectations']
s_lowrank=expectations_lowrank['expectations']
s_lowrank2=expectations_lowrank2['expectations']

print('exact runtime is {:.2f} seconds'.format(expectations_exact['runtime']))
print('monte carlo runtime is {:.2f} seconds'.format(expectations_mc['runtime']))
print('low-rank runtime is {:.2f} seconds'.format(expectations_lowrank['runtime']))


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
    # plt.plot(tlist, np.real(s_lowrank2[:,n]), color='green', linestyle='-.')
plt.plot(tlist, np.real(s_exact[N-1]), color='red', label='exact $\\langle\\sigma_z^{(j)}\\rangle$' )
# plt.plot(tlist, np.real(s_lowrank[:, N-1]), color='blue',   linestyle = '--', label = 'rank %i $\\langle\\sigma_z^{(j)}\\rangle$' % rank )
plt.plot(tlist, np.real(s_lowrank[:, N-1]), color='blue',   linestyle = '--', label = 'low-rank $\\langle\\sigma_z^{(j)}\\rangle$'  )

# plt.plot(tlist, np.real(s_lowrank2[:, N-1]), color='green',   linestyle = '-.',label = 'rank %i $\\langle\\sigma_z^{(j)}\\rangle$' % rank2 )

# plt.xlabel('Time')
plt.ylabel('$\\langle\sigma^{(j)}_z\\rangle$')
plt.legend()
plt.grid(True)
# plt.xlim(0,95)

plt.subplot(212)
for n in range(N-1):
    plt.plot(tlist, np.real(s_exact[n]), color='red')
    plt.plot(tlist, np.real(s_mc[n]), color='blue', linestyle='--')
plt.plot(tlist, np.real(s_exact[N-1]), color='red', label='exact $\\langle\\sigma_z^{(j)}\\rangle$' )
plt.plot(tlist, np.real(s_mc[N-1]), color='blue',   linestyle = '--', label = 'Monte-Carlo $\\langle\\sigma_z^{(j)}\\rangle$' )
plt.legend()
plt.ylabel('$\\langle\sigma^{(j)}_z\\rangle$')
plt.xlabel('Time')
plt.grid(True)
plt.savefig('./Plots/lowrankvsmontecarlo' + fig_params,bbox_inches='tight')
plt.show()


outfile_exact = './Data/exact:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu.npz'.format(
    N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving)
expectations_exact = np.load(outfile_exact)
cmap = plt.get_cmap('jet_r')

for factor in [4]:
    color = cmap((float(factor)**2 +0.4)/20)
    bath_couple=10**(-factor)
    if factor==0.5:
        bath_couple = 5e-3
    outfile_exact = './Data/exact:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu.npz'.format(
        N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving)
    expectations_exact = np.load(outfile_exact)
    s_exact = expectations_exact['expectations']
    # s_exact=expectations_mc['expectations']

    mc_times=[]
    lr_times=[]
    mc_eps=[]
    lr_eps=[]
    for ntraj in [100,1000,5000]:
        outfile_mc = './Data/montecarlo:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu-{}ntraj.npz'.format(
            N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving, ntraj)
        expectations_mc = np.load(outfile_mc)

        s_mc = expectations_mc['expectations']
        mc_times.append(expectations_mc['runtime'])
        eps_z_mc=0
        for n in range(N):
            eps_z_mc+= np.sum((s_exact[n]-s_mc[n])**2)/np.sum(s_exact[n]**2)
        mc_eps.append(np.sqrt(eps_z_mc))
        # print(mc_eps)color = cmap((float(xx)-7)/45)
    for rank in [1,2,4,8]:
        outfile_lowrank = './Data/lowrank:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu-{}rank.npz'.format(
            N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving, rank)
        expectations_lowrank = np.load(outfile_lowrank)
        s_lowrank = expectations_lowrank['expectations']
        lr_times.append(expectations_lowrank['runtime'])
        eps_z_lr = 0
        for n in range(N):
            eps_z_lr += np.sum((s_exact[n] - s_lowrank[:, n]) ** 2)/np.sum(s_exact[n]**2)
        lr_eps.append(np.sqrt(eps_z_lr))
    #
    # if factor==2:
    #     rank=8
    #     outfile_lowrank = './Data/lowrank:{}sites-init{}-{}h-{}Jx-{}Jy-{}Jz-{}t_max-{}steps-{}dephase-gamma{}-{}mu-{}rank.npz'.format(
    #         N, init, h_param, Jx_param, Jy_param, Jz_param, endtime, steps, dephase, bath_couple, driving, rank)
    #     expectations_lowrank = np.load(outfile_lowrank)
    #     s_lowrank = expectations_lowrank['expectations']
    #     lr_times.append(expectations_lowrank['runtime'])
    #     eps_z_lr = 0
    #     for n in range(N):
    #         eps_z_lr += np.sum((s_exact[n] - s_lowrank[:, n]) ** 2)/np.sum(s_exact[n]**2)
    #     lr_eps.append(np.sqrt(eps_z_lr))
    #     print(lr_times)
    #     print(lr_eps)
    #
    plt.plot(lr_times,lr_eps,color=color,marker="o",label='low-rank $\\Gamma=10^{-%d}$' % factor)
    plt.xlabel('Runtime(s)')
    plt.grid(True)
    plt.ylabel('Error')
    # plt.plot(mc_times,mc_eps,color=color,linestyle='dashed',marker="^",label='Monte-Carlo  $\\Gamma=10^{-%d}$' % factor)
    plt.loglog(mc_times,mc_eps,color=color,linestyle='dashed',marker="^",label='Monte-Carlo  $\\Gamma=10^{-%d}$' % factor)
    plt.legend()

plt.savefig('./Plots/erroslowrankvsmontecarlo' + fig_params,bbox_inches='tight')
plt.show()
