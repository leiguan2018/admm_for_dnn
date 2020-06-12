import cupy as np
import torch
import torch.distributed as dist
from cupy import matmul as mul
from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack


# return softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
# return cross entropy
def cross_entropy(label, prob):
    loss = -np.sum(label * np.log(prob))
    return loss
# return the cross entropy loss function
def cross_entropy_with_softmax(label, z):
    prob = softmax(z)
    loss = cross_entropy(label, prob)
    return loss
#return the  relu function
def relu(x):
    return np.maximum(x, 0)


##############################
# update W
#################################
def update_W(W, a_last, z, u, rho):
    size = dist.get_world_size()
    rank = dist.get_rank()
    # convert to pytorch data

    #update W
    temp1 = z + u/rho

    temp1 = from_dlpack(toDlpack(temp1))
    a_last = from_dlpack(toDlpack(a_last))

    data1 = torch.mm(temp1, torch.t(a_last))
    data2 = torch.mm(a_last, torch.t(a_last))
    data = torch.cat((data1, data2), 0)
    # data = comm.reduce(data, op=MPI.SUM, root=0)
    dist.reduce(data, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        middle_pos = data1.shape[0]
        data1 = data[0: middle_pos]
        data2 = data[middle_pos:]
        inverse_data = torch.pinverse(data2)
        W = torch.mm(data1, inverse_data)
    else:
        W = from_dlpack(toDlpack(W))
        # W = None
    dist.broadcast(W, src=0)

    # convert to cupy data
    W = fromDlpack(to_dlpack(W))
    return W


##############################
# update a
#################################
# return the objective value of a-subproblem
def obj_a(W_next, a, z, z_next, u_next, rho, gamma):
    temp = z_next - mul(W_next, a) +u_next/rho
    res = rho / 2 * np.sum(temp * temp)
    #return res
    res = res + gamma/2*np.sum((a-relu(z))*(a-relu(z)))
    return res

# return the gradient of a-subproblem
def grad_a(W_next, a, z, z_next, u_next,rho, gamma):
    res = rho * mul(np.transpose(W_next), mul(W_next, a) - z_next-u_next/rho)
    res = res + gamma*(a-relu(z))
    return res

# return the result of a-subproblem
def update_a(W_next, a_old, z_next, z, u_next, rho, gamma):
    obj = 10e10
    gradient = grad_a(W_next, a_old, z, z_next, u_next, rho, gamma)
    t=1
    eta = 2
    a_new = a_old-gradient/t
    MAX_ITER = 500
    TOLERANCE = 1e-3
    for i in range(MAX_ITER):
        obj_old = obj
        obj = obj_a(W_next, a_new, z, z_next, u_next, rho, gamma)
        if abs(obj-obj_old) < TOLERANCE:
            break
        t = t * eta
        #gradient = a_obj_gradient(beta, W_next, b_next, z_next, u_next, v,z,rho)
        a_new =a_old-gradient/t
    a = a_new
    return a


##############################
# update z
#################################
# return the result of z-subproblem
def update_z(W, a_last, a, z, rho, gamma):
    m = mul(W, a_last)
    sol1 = (gamma*a + rho*m)/(gamma + rho)
    sol2 = m
    z1 = np.zeros_like(a)
    z2 = np.zeros_like(a)
    z  = np.zeros_like(a)

    z1[sol1>=0.] = sol1[sol1>=0.]
    z2[sol2<=0.] = sol2[sol2<=0.]
    #print("z1, z2", sol1>=0., sol1[sol1>=0.])

    fz_1 = gamma * np.power(a - relu(z1), 2) + rho * np.power(z1 - m, 2)
    fz_2 = gamma * np.power(a - relu(z2), 2) + rho * np.power(z2 - m, 2)

    index_z1 = fz_1<=fz_2
    index_z2 = fz_2<fz_1
    z[index_z1] = z1[index_z1]
    z[index_z2] = z2[index_z2]
    return z

# return the result of z_L-subproblem by FISTA
def update_zl(W, a_last, zl_old, label, u, rho):
    fzl = 10e10
    MAX_ITER = 500
    zl = zl_old
    t = 1
    zeta = zl
    eta = 4
    TOLERANCE = 1e-3
    for i in range(MAX_ITER):
        fzl_old = fzl
        fzl = cross_entropy_with_softmax(label, zl)+rho/2*np.sum((zl-mul(W,a_last)+u/rho)*(zl-mul(W,a_last)+u/rho))
        if abs(fzl - fzl_old) < TOLERANCE:
            break
        t_old = t
        t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        theta = (1 - t_old) / t
        gradients2 = (softmax(zl) - label)
        zeta_old = zeta
        zeta = (rho * (mul(W, a_last)-u/rho) + (zl - eta * gradients2) / eta) / (rho + 1 / eta)
        zl = (1 - theta) * zeta + theta * zeta_old
    return zl