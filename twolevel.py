from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
default_times = np.linspace(-50, 50, 10000)


def evolution_LZ(delta=1, velocity=1, gamma1=0, gamma2=0, rho=fock_dm(2, 0), times=default_times):
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sp = sigmap()
    sm = sigmam()

    c_op_list = []
    if gamma1 > 0.0:
        c_op_list.append(np.sqrt(gamma1) * sz)
    if gamma2 > 0.0:
        c_op_list.append(np.sqrt(gamma2) * sm)

    H0 = delta * sx
    H = [H0, [velocity * sz, 't']]
    result = mesolve(H, rho_0, times, c_op_list, e_ops=[sx, sz], progress_bar=True)

    return result.expect


# velocity=0.005
# delta=1
# gamma1=0.000
# gamma2=0.000

# times=default_times
# lzlimit=np.exp(-np.pi*delta**2/velocity)
# szlimit=(2*np.exp(-np.pi*pow(delta,2)/(velocity))-1)
# expect=evolution_LZ(delta,velocity,gamma1,gamma2,rho_0,times)
#
# # plt.plot(times,expect[0])
# plt.plot(times,expect[1])
# plt.plot(times,np.ones(len(times))*szlimit)
# plt.show()

def evolution(delta=1, epsilon=1, gamma1=0, gamma2=0, rho=fock_dm(2, 0), times=default_times):
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sp = sigmap()
    sm = sigmam()

    c_op_list = []
    if gamma1 > 0.0:
        c_op_list.append(np.sqrt(gamma1) * sz)
    if gamma2 > 0.0:
        c_op_list.append(np.sqrt(gamma2) * sm)

    H = delta * sx + epsilon * sz
    result = mesolve(H, rho_0, times, c_op_list, e_ops=[sx, sz], progress_bar=True)
    print(result)
    return result.expect


# epsilon=1
# delta=1
# gamma1=1
# gamma2=0
# rho_0=fock_dm(2,0)
# psi_0=basis(2,0)
# times=np.linspace(0,0.01,2)
# expect=evolution(delta,epsilon,gamma1,gamma2,rho_0,times)
#
# plt.plot(times,expect[0])
# plt.plot(times,expect[1])
# # plt.plot(times,np.ones(len(times))*szlimit)
# plt.show()

def U_prop_sz(psi, epsilon=1, delta=1, gamma=0, K=1, dt=10 ** -5):
    sx = sigmax()
    sz = sigmaz()

    H0 = epsilon * sz + delta * sx
    HI = -1j * sz * np.sqrt(K * gamma * dt)
    H = 1j * H0 * dt + HI
    prop_U = H.expm()
    result = prop_U * psi
    return result


def V_prop_sz(psi, epsilon=1, delta=1, gamma=0, K=1, dt=10 ** -5):
    sx = sigmax()
    sz = sigmaz()

    H0 = epsilon * sz + delta * sx
    HI = +1j * sz * np.sqrt(K * gamma * dt)
    H = 1j * H0 * dt + HI
    prop_V = H.expm()
    result = prop_V * psi
    # print('propagator result')
    # print(result)
    return result


def U_prop_sm(psi, epsilon=1, delta=1, gamma=0, K=1, dt=10 ** -5):
    sx = sigmax()
    sz = sigmaz()
    sm = sigmam()
    H0 = epsilon * sz + delta * sx - 1j * gamma * (K / 2) * (sm * sm - sm.dag() * sm)
    HI = -1j * sm * np.sqrt(K * gamma * dt)
    H = 1j * H0 * dt + HI
    prop_U = H.expm()
    result = prop_U * psi
    return result


def V_prop_sm(psi, epsilon=1, delta=1, gamma=0, K=1, dt=10 ** -5):
    sx = sigmax()
    sz = sigmaz()
    sm = sigmam()
    H0 = epsilon * sz + delta * sx - 1j * gamma * (K / 2) * (sm * sm - sm.dag() * sm)

    HI = +1j * sm * np.sqrt(K * gamma * dt)
    H = 1j * H0 * dt + HI
    prop_V = H.expm()
    result = prop_V * psi
    # print('propagator result')
    # print(result)
    return result


def one_step(psi_list, epsilon=1, delta=1, gamma=0, dt=10 ** -5):
    psis = []
    for psi in psi_list:
        psis.append(U_prop_sz(psi, epsilon, delta, gamma, 1, dt))
        psis.append(V_prop_sz(psi, epsilon, delta, gamma, 1, dt))
    # print('last psi')
    return psis


def one_step_multiple(psi_list, epsilon=1, delta=1, gamma1=0, gamma2=0, dt=10 ** -5):
    psis = []
    K = 1
    if gamma1 > 0 and gamma2 > 0:
        K = 2
    prefactor = np.sqrt(1 / (2 * K))
    for psi in psi_list:
        if gamma1 > 0:
            psis.append(prefactor * U_prop_sz(psi, epsilon, delta, gamma1, K, dt))
            psis.append(prefactor * V_prop_sz(psi, epsilon, delta, gamma1, K, dt))
        if gamma2 > 0:
            psis.append(prefactor * U_prop_sm(psi, epsilon, delta, gamma2, K, dt))
            psis.append(prefactor * V_prop_sm(psi, epsilon, delta, gamma2, K, dt))
        else:
            psis.append(U_prop_sz(psi, epsilon, delta, 0, K, dt))

    # print('last psi')
    return psis


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


def orthogonalise(psi_list, U, rank):
    g_list = []
    U = U.T
    if rank > len(psi_list):
        rank = len(psi_list)
    for i in range(rank):
        norm = 0
        g = Qobj([[0], [0]])
        for j in range(len(psi_list)):
            # print(U[i,j])
            # print(psi_list[j].dims)
            g_unit = U[i, j] * psi_list[j]
            g += g_unit
        g_list.append(g)

    return g_list


def rho(psi_list):
    zeroket = Qobj([[0], [0]])
    # print(zeroket)
    rho = ket2dm(zeroket)
    # print(rho)
    for psi in psi_list:
        # print('psi is')
        # print(psi.unit())
        # rhoj=ket2dm(psi)
        # print(rhoj)
        rho += ket2dm(psi)
        # rho=rho.unit()
    # print('rho is')
    # print(rho)

    # print(rho.type)
    return rho


dt = 10 ** -5
epsilon = 1
delta = 1
gamma = 0.1
gamma1 = gamma
gamma2 = 0.2
rho_0 = fock_dm(2, 0)
psi_0 = basis(2, 0)
psis = one_step([psi_0], epsilon, delta, gamma, dt)
print(psis[0])
print(psis[1])

N = 1000
times, dt = np.linspace(0, 25, N, retstep=True)
print("timestep")
print(dt)

sz = sigmaz()
sx = sigmax()

# z_track.append((sz * rho(psi_list)).tr())
expect = evolution(delta, epsilon, gamma1, gamma2, rho_0, times)
expect_unaltered = evolution(delta, epsilon, 0, 0, rho_0, times)

for rank in [1, 2]:
    psi_list = [psi_0]
    z_track = []
    x_track = []
    for k in tqdm(range(0, N)):
        z_track.append((sz * rho(psi_list)).tr())
        x_track.append((sx * rho(psi_list)).tr())
        # psi_list=one_step(psi_list,epsilon,delta,gamma,dt)
        psi_list = one_step_multiple(psi_list, epsilon, delta, gamma1, gamma2, dt)

        if len(psi_list) > rank:
            mat = overlap(psi_list)
            w, v = np.linalg.eig(mat)
            idx = w.argsort()[::-1]
            w = w[idx]
            v = v[:, idx]
            psi_list = orthogonalise(psi_list, v, rank)
        # norms=[psi.norm() for psi in psi_list]
        # norm=np.sum(norms)
        # psi_list = [psi/norm for psi in psi_list]
    # mat = overlap(psi_list)
    # fig, ax = plt.subplots()
    # # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    # ax.matshow(mat.real, cmap='seismic')
    #
    # for (i, j), z in np.ndenumerate(mat.real):
    #     ax.text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
    plt.subplot(211)
    plt.plot(times, z_track, label='rank %s' % rank)

    plt.subplot(212)
    plt.plot(times, x_track, label='rank %s' % rank)

plt.subplot(211)
plt.plot(times, expect[1], label='Lindblad', color='black', linestyle='--')
plt.plot(times, expect_unaltered[1], label='Dissipation free', linestyle='-.', color='gray')
# plt.xlabel('Time')
plt.ylabel('$\\langle S_Z\\rangle$')

plt.subplot(212)
plt.plot(times, expect[0], label='Lindblad', color='black', linestyle='--')
plt.plot(times, expect_unaltered[0], label='dissipation free', linestyle='-.', color='gray')
plt.xlabel('Time')
plt.ylabel('$\\langle S_X\\rangle$')
plt.legend()
plt.show()
