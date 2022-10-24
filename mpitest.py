from mpi4py import MPI

comm = MPI.COMM_WORLD

print("rank={}/{}".format(comm.Get_rank(), comm.Get_size()))