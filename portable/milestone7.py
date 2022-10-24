import time
import sys
import os
import csv
import glob
import numpy as np
from mpi4py import MPI
np.set_printoptions(threshold=sys.maxsize)


# --- Constants -----------------------------------------------------------------------------------
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], 
              [0, 0, 1, 0, -1, 1, 1, -1, -1]])
#c_op = c[[1,0]]

# --- Functions -----------------------------------------------------------------------------------
calc_rho = lambda f: np.sum(f, axis=0)
calc_vel = lambda f, rho: np.dot(f.T, c[[1,0]].T).T / rho
calc_neu = lambda omega: (1/3 * (1/omega - 1/2)) 
calc_reynolds = lambda omega, length, vel: (vel * length) / calc_neu(omega)


def f_stream(f):

    for i in range(9):
        f[i,:,:] = np.roll(f[i,:,:], c[:,i], axis=(1,0)) 

    return f                                                                     

def f_equilibrium(rho, vel):

    var1 = np.transpose(vel, (1,0,2))
    var2 = np.transpose(np.dot(c[[1,0]].T, var1), (0,1,2))
    var3 = vel[0,:,:]**2 + vel[1,:,:]**2
    var4 = np.array([rho * w[i] for i in range(9)])

    return (var4 * (1 + 3 * var2 + 9/2 * var2**2 - 3/2 * var3))
    
def f_collision(f, omega):

    rho = calc_rho(f)
    vel = calc_vel(f, rho)
    f_eq = f_equilibrium(rho,vel)
    f = f + omega * (f_eq - f)

    return f, rho, vel, f_eq

def f_moving_wall(f, lid_vel):
    w_rho = 2 * (f[7,-2, :] + f[4,-2, :] + f[8,-2, :]) + f[3,-2, :] + f[0,-2, :] + f[1,-2, :]

    f[4, -2,  :] = f[2, -1,  :]
    f[8, -2,  :] = f[6, -1,  :] + 6 * w[6] * lid_vel * w_rho
    f[7 ,-2,  :] = f[5, -1,  :] - 6 * w[5] * lid_vel * w_rho

    return f
    
def f_rigid_wall(f, top, down, left, right):

    if top:
        f[[7,4,8],-2,:] = f[[5,2,6],-1,:]
    if down:
        f[[5,2,6],1,:] = f[[7,4,8],0,:]
    if left:
        f[[8,1,5],:,1] = f[[6,3,7],:,0]
    if right:
        f[[7,3,6],:,-2] = f[[5,1,8],:,-1]

    return f



def save_mpiio(comm, fn, g_kl):
    """
    Write a global two-dimensional array to a single file in the npy format
    using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html

    Arrays written with this function can be read with numpy.load.

    Parameters
    ----------
    comm
        MPI communicator.
    fn : str
        File name.
    g_kl : array_like
        Portion of the array on this MPI processes. This needs to be a
        two-dimensional array.
    """
    from numpy.lib.format import dtype_to_descr, magic
    magic_str = magic(1, 0)

    local_nx, local_ny = g_kl.shape
    nx = np.empty_like(local_nx)
    ny = np.empty_like(local_ny)

    commx = comm.Sub((True, False))
    commy = comm.Sub((False, True))
    commx.Allreduce(np.asarray(local_nx), nx)
    commy.Allreduce(np.asarray(local_ny), ny)

    arr_dict_str = str({ 'descr': dtype_to_descr(g_kl.dtype),
                         'fortran_order': False,
                         'shape': (np.asscalar(nx), np.asscalar(ny)) })
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny*local_nx), offsetx)
    offsety = np.zeros_like(local_ny)
    commy.Exscan(np.asarray(local_ny), offsety)

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode('latin-1'))
    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety+offsetx)*mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()

def f_communicator(f, commCart):
   
    top_src, top_dst = commCart.Shift(0, -1)
    bot_src, bot_dst = commCart.Shift(0, +1)
    lef_src, lef_dst = commCart.Shift(1, -1)
    rig_src, rig_dst = commCart.Shift(1, +1)

    # for 4 in
    '''
    6 7 8
    3 4 5
    0 1 2 <- moving lid here
    '''
    
    p1 = f[:,  1,  :].copy()
    p2 = f[:, -1,  :].copy()
    commCart.Sendrecv(p1, top_dst, recvbuf=p2, source=top_src) 
    f[:, -1,  :] = p2
    
    p1 = f[:, -2,  :].copy()
    p2 = f[:,  0,  :].copy()
    commCart.Sendrecv(p1, bot_dst, recvbuf=p2, source=bot_src) 
    f[:,  0,  :] = p2
    
    p1 = f[:,  :,  1].copy()
    p2 = f[:,  :, -1].copy()
    commCart.Sendrecv(p1, lef_dst, recvbuf=p2, source=lef_src) 
    f[:,  :, -1] = p2
    
    p1 = f[:,  :, -2].copy()
    p2 = f[:,  :,  0].copy()
    commCart.Sendrecv(p1, rig_dst, recvbuf=p2, source=rig_src)
    f[:,  :,  0] = p2
    
    return f

def f_wall_parallel(f, coords):
   
    if (coords[0] == 0):
        f = f_rigid_wall(f, False, True, False, False)

    if (coords[0] == (y_decomp - 1)):
        f = f_moving_wall(f, lid_vel)
        
    if (coords[1] == 0):
        f = f_rigid_wall(f, False, False, True, False)
        
    if (coords[1] == (x_decomp - 1)):
        f = f_rigid_wall(f, False, False, False, True)
    
    return f

# ---
steps = int(sys.argv[1])
recording_steps = int(sys.argv[2])
length = int(sys.argv[3])
width = int(sys.argv[4])
omega = float(sys.argv[5])
lid_vel = float(sys.argv[6])
y_decomp = int(sys.argv[7])
x_decomp = int(sys.argv[8])
#print(sys.argv[1:])
# ---

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
CommCart = comm.Create_cart((y_decomp, x_decomp), periods=(False, False), reorder=False)
coords = CommCart.Get_coords(rank)


rows = (width // y_decomp) + 2
columns = (length // x_decomp) + 2

'''
rows, columns = add_drynodes(coords, rows, columns)
'''

rho = np.ones((rows, columns), dtype=np.float64)
vel = np.zeros((2,rows, columns), dtype=np.float64)

f = f_equilibrium(rho, vel)
'''
files = glob.glob('data/*')
for j in files:
    try:
        os.remove(j)
    except:
        pass
'''
for i in range(steps):
    f = f_communicator(f,CommCart)
    f = f_stream(f)
#   f = f_boundary(f, False, True, True, True)
#   f = f_moving_lid(f, lid_vel)
    f = f_wall_parallel(f, coords)
    f, rho, vel, f_eq = f_collision(f, omega)
    if i%recording_steps==0:
        save_mpiio(CommCart, "data/grid_{}_decomp_{}_uy_{}.npy".format(length, x_decomp, i), vel[0,1:-1,1:-1])
        save_mpiio(CommCart, "data/grid_{}_decomp_{}_ux_{}.npy".format(length, x_decomp, i), vel[1,1:-1,1:-1])





