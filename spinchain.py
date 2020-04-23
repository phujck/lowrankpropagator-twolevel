from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

params = {
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [2 * 3.375, 2 * 3.375],
    'text.usetex': True,
    'figure.figsize': (16, 12)
}

plt.rcParams.update(params)


def integrate_setup(N, h, Jx, Jy, Jz, gamma, leads):
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sp = sigmap()
    sm = sigmam()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += - 0.5 * h[n] * sz_list[n]

    # interaction terms
    for n in range(N - 1):
        H += - 0.5 * Jx[n] * sx_list[n] * sx_list[n + 1]
        H += - 0.5 * Jy[n] * sy_list[n] * sy_list[n + 1]
        H += - 0.5 * Jz[n] * sz_list[n] * sz_list[n + 1]

    # collapse operators
    c_op_list = []

    # spin dephasing
    for n in range(N):
        if gamma[n] > 0.0:
            c_op_list.append(np.sqrt(gamma[n]) * sz_list[n])

    # adding coupling to bath:
    if leads[0] > 0:
        mu = leads[1]
        gamma_l = leads[0]
        op_list = []
        for m in range(N):
            op_list.append(si)
        zero_minus = op_list.copy()
        zero_plus = op_list.copy()
        N_minus = op_list.copy()
        N_plus = op_list.copy()
        zero_minus[0] = sm * np.sqrt(gamma_l * (1 + mu))
        zero_plus[0] = sp * np.sqrt(gamma_l * (1 - mu))
        N_minus[-1] = sm * np.sqrt(gamma_l * (1 - mu))
        N_plus[-1] = sp * np.sqrt(gamma_l * (1 + mu))
        c_op_list.append(tensor(zero_minus))
        # c_op_list.append(tensor(zero_plus))
        # c_op_list.append(tensor(N_minus))
        c_op_list.append(tensor(N_plus))
    return H, sz_list, sx_list, c_op_list


def qutip_solver(H, psi0, tlist, c_op_list, sz_list, solver):
    # evolve and calculate expectation values
    if solver == "me":
        result = mesolve(H, psi0, tlist, c_op_list, sz_list, progress_bar=True)
    elif solver == "mc":
        ntraj = 250
        result = mcsolve(H, psi0, tlist, c_op_list, sz_list, ntraj, progress_bar=True)

    return result.expect


def unitaries(H, c_op_list, dt):
    K = len(c_op_list)
    U_ops = []
    if K==0:
        U_ops.append((1j*H*dt).expm())
    else:
        prefactor = np.sqrt(1 / (2 * K))
        for op in c_op_list:
            exponent = 1j * H * dt
            if op.isherm == False:
                exponent += (dt * K / 2) * (op * op - op.dag() * op)
            extra = 1j * np.sqrt(K * dt) * op
            P1 = exponent + extra
            P2 = exponent - extra
            U_prop=prefactor*(P1.expm())
            V_prop=prefactor*(P2.expm())
            U_ops.append(U_prop)
            U_ops.append(V_prop)
    return U_ops

def overlap(psi_list):
    matrix = np.zeros((len(psi_list), len(psi_list)), dtype=np.complex_)
    for i in range(len(psi_list)):
        for j in range(i, len(psi_list)):
            # overlap=psi_list[i].dag()*psi_list[j]
            # print('method 1')
            # print(overlap.data)
            # print('method 2')
            o2 = psi_list[i].overlap(psi_list[j])
            # print(o2)
            matrix[i, j] = o2
            if j > i:
                matrix[j, i] = np.conj(o2)
    # print(matrix)
    return matrix


def orthogonalise(psi_list, U, zeroobj, rank):
    g_list = []
    U = U.T
    if rank > len(psi_list):
        rank = len(psi_list)
    norm = 0
    for i in range(rank):
        g = zeroobj
        for j in range(len(psi_list)):
            # print(U[i,j])
            # print(psi_list[j].dims)
            g_unit = U[i, j] * psi_list[j]
            g += g_unit
        norm+=g.norm()**2
        g_list.append(g)
    g_list=[g/np.sqrt(norm) for g in g_list]
    return g_list

def one_step(psis,U_ops,zeropsi,rank):
    new_psis=[]
    for psi in psis:
        for U in U_ops:
            newpsi=U*psi
            new_psis.append(newpsi)
    if len(new_psis) > rank:
        mat = overlap(new_psis)
        w, v = np.linalg.eig(mat)
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:, idx]
        new_psis = orthogonalise(new_psis, v, zeropsi, rank)
    return new_psis

def expectations(psis,ops,normalise):
    expecs=[]
    for op in ops:
        expec=0
        norm=0
        for psi in psis:
            expec+=expect(op,psi)
            if normalise:
                norm+=psi.norm()
        if normalise:
            expec=expec/norm
        expecs.append(expec)
    return expecs

# set up the calculation
#
# solver = "me"  # use the ode solver
solver = "mc"   # use the monte-carlo solver

N = 12  # number of spins

# array of spin energy splittings and coupling strengths. here we use
# uniform parameters, but in general we don't have too
h = 0.1 * 2 * np.pi * np.ones(N)
Jz = 0.1 * 2 * np.pi * np.ones(N)
Jx = 0.1 * 2 * np.pi * np.ones(N)
# Jx = 0 * 2 * np.pi * np.ones(N)
Jy = 0.1 * 2 * np.pi * np.ones(N)
# dephasing rate
# gamma = 0.01 * np.ones(N)
dephase = 0
gamma = dephase * np.ones(N)

# leads coupling
leads = np.zeros(2)
# bath coupling
bath_couple = 0.1
driving = 1
leads[0] = bath_couple
# driving
leads[1] = driving
# driving
# intial state, first spin in state |1>, the rest in state |0>
psi_list = []
zero_list=[]
zero_ket=Qobj([[0], [0]])
# psi_list.append(basis(2,1))
# for n in range(N-1):
#     psi_list.append(basis(2,0))

start=time.time()
for n in range(N):
    psi_list.append((basis(2, 0) + basis(2, 1)).unit())
    zero_list.append(zero_ket)

# for n in range(N):
#     if n %2 ==0:
#         psi_list.append(basis(2,0))
#     else:
#         psi_list.append(basis(2,1))
psi0 = tensor(psi_list)
zeropsi=tensor(zero_list)
# print(zeropsi)
steps = 1000
tlist, deltat = np.linspace(0, 20, steps, retstep=True)
rank=2
H, sz_list, sx_list, c_op_list = integrate_setup(N, h, Jx, Jy, Jz, gamma, leads)
U_ops= unitaries(H,c_op_list,deltat)
end=time.time()
print('operators built! Time taken %.3f seconds' % (end-start))
psis=[psi0]

# visualise sparsity
# loop_start=time.time()
# for j in range(0,50):
#     psis=one_step(psis,U_ops,zeropsi,rank)
#     newend = time.time()
#     start=time.time()
#     norms=[psi.norm()**2 for psi in psis]
#     norm=np.sum(norms)
#     psi_list = [psi/np.sqrt(norm) for psi in psis]
#     end=time.time()
#     print('Normalisation done. Time taken %.5f seconds' % (end-start))
# loop_end=time.time()
# # psi_list = [psi for psi in psis]
# print('50 steps done. Time taken %.5f seconds' % (loop_end - loop_start))

# mat = overlap(psi_list)
# print(np.sum(np.diag(mat)))
# fig, ax = plt.subplots()
# ax.matshow(mat.real, cmap='seismic')
#
# for (i, j), z in np.ndenumerate(mat.real):
#     ax.text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
#
#
# # plt.spy(mat,markersize=2,precision=10**(-6))
#
# plt.show()


# print(psis)


sz_expt = qutip_solver(H, psi0, tlist, c_op_list, sz_list+sx_list, solver)
loop_start=time.time()
s_rank=[]
psi_times=[]
time_psis=[]
for j in tqdm(range(steps)):
    # s_rank.append(expectations(psis,sz_list+sx_list))
    time_psis.append(psis)
    psis=one_step(psis,U_ops,zeropsi,rank)
    # norms=[psi.norm()**2 for psi in psis]
    # norm=np.sum(norms)
    # psis = [psi/np.sqrt(norm) for psi in psis]
loop_end=time.time()
# psi_list = [psi for psi in psis]
print('Low rank propagation done. Time taken %.5f seconds' % (loop_end - loop_start))

for psis in time_psis:
    s_rank.append(expectations(psis,sz_list+sx_list,False))
s_rank=np.array(s_rank)
print(s_rank.size)
# print(np.array(s_rank))
print(s_rank[:,0])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
sz_tot = np.zeros(len(sz_expt[0]))
for n in range(N):
    sz_tot += sz_expt[n]

for n in range(N):
    ax1.plot(tlist, np.real(sz_expt[n]), label='$\\langle\\sigma_z^{(%d)}\\rangle$' % n)
    ax1.plot(tlist, np.real(s_rank[:,n]), linestyle='--',label='Low-rank $\\langle\\sigma_z^{(%d)}\\rangle$' % n)
    ax2.plot(tlist, np.real(sz_expt[N + n]), label='$\\langle\\sigma_x^{(%d)}\\rangle$' % n)
    ax2.plot(tlist, np.real(s_rank[:,N + n]),linestyle='--', label='Low-rank $\\langle\\sigma_x^{(%d)}\\rangle$' % n)
ax1.legend(loc=0)
ax2.set_xlabel('Time [ns]')
ax1.set_ylabel('$\\langle\sigma_z\\rangle$')
ax2.set_ylabel('$\\langle\sigma_x\\rangle$')
ax2.legend()

ax1.set_title('Dynamics of a Heisenberg spin chain')
plt.show()
#
# plt.plot(tlist, sz_tot)
# plt.xlabel('Time [ns]')
# plt.ylabel('$\\langle\sigma^{\\rm tot}_z\\rangle$')
# plt.title('Dynamics of a Heisenberg spin chain')
# plt.show()
#
# final_sz = [sz_expt[n][-1] for n in range(N)]
# plt.plot(range(N), final_sz)
# plt.show()
