import numpy as np
import matplotlib.pyplot as plt
import imageio
import matplotlib.cm as cm
from mpi4py import MPI
import time

name = 'animation_walking.gif'


def plot_fun(set, name_pict):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(set, cmap=cm.gray)
    plt.savefig("fig/"+name_pict+".png")


def anim(name_pict):
    with imageio.get_writer(name, duration=0.5) as writer:
        writer.append_data(imageio.imread(name_pict))


comm = MPI.COMM_WORLD
size_x, size_y = 10, 10
coeff = 0.9
s = comm.size

n = size_y//s
n_last = size_y - n*s
rank = comm.rank

comm.Barrier()

if comm.rank != s-1:
    part_picture = np.random.choice([0, 1], (size_x, n+2), replace=True, p=[coeff, 1.0 - coeff])
    print('DOING rand_picture')
    if rank != 0:
        comm.isend(part_picture, dest=0)#, tag=comm.rank)
        print('Process {} sent data:'.format(rank), part_picture)

else:
    part_picture = np.random.choice([0, 1], (size_x, n+n_last+2), replace=True, p=[coeff, 1.0 - coeff])
    print('DOING rand_picture')

    comm.isend(part_picture, dest=0)#, tag=comm.rank)
    print('Process {} sent data:'.format(rank), part_picture)

comm.Barrier()

if comm.rank == 0:
    picture = np.zeros((size_x, size_y))
    picture[:, 0:n] = part_picture[:, 1: -1]

    for i in range(1, s):
        request = comm.irecv(source=i)#, tag=i)
        recvdata = request.wait()
        print('Process {} recvdata[:, 1:-1]] data:'.format(rank))
        print(recvdata[:, 1:-1])
        print("This is picture[:, i*n:i*n+n]")
        print(picture[:, i*n:i*n+n])

        if i == s-1:
            picture[:, (size_y-n-n_last):size_y] = recvdata[:, 1:-1]
        else:
            picture[:, i*n:i*n+n] = recvdata[:, 1:-1]

    plot_fun(picture, "foo")

comm.Barrier()
#for i in range(s):
#    comm.Barrier()

# loop

if comm.rank != s-1:
    for i in range(size_x):
        part_picture[i] = np.roll(part_picture[i], 1)
    print('ROLL part_picture')

    if rank != 0:
        comm.isend(part_picture[:, -1], dest=comm.rank+1)#, tag=comm.rank)
        print('Process {} sent ROLL data:'.format(rank), part_picture[:, -1])

else:
    for i in range(size_x):
        part_picture[i] = np.roll(part_picture[i], 1)
    print('ROLL part_picture')

    comm.isend(part_picture[:, -1], dest=0)#, tag=comm.rank)
    print('Process {} sent ROLL data:'.format(rank), part_picture[:, -1])

comm.Barrier()

#for i in range(s):
#    comm.Barrier()


# combine the picture
if comm.rank == 0:
    request1 = comm.irecv(source=s-1)#, tag=s-1)
    recvdata1 = request1.wait()
    part_picture[:, 1] = recvdata1

else:
    request1 = comm.irecv(source=comm.rank-1)#, tag=comm.rank-1)
    recvdata1 = request1.wait()
    part_picture[:, 1] = recvdata1
    comm.isend(part_picture[:, 1: -1], dest=0)#, tag=comm.rank+s)
    print('Process {} sent ROLL data for picture:'.format(rank), part_picture[:, 1: -1])

'''
comm.Barrier()

if comm.rank != 0:
    # send to create picture

    comm.isend(part_picture[:, 1: -1], dest=0)#, tag=comm.rank+s)
    print('Process {} sent ROLL data for picture:'.format(rank), part_picture[:, 1: -1])

# comm.Barrier()




comm.Barrier()
#for i in range(s):
#    comm.Barrier()
# create picture

if comm.rank == 0:
    picture = np.zeros((size_x, size_y))
    picture[:, 0:n] = part_picture[:, 1: -1]

    for i in range(1, s):
        request2 = comm.irecv(source=i)#, tag=i+s)
        recvdata2 = request2.wait()
        print('Process {} recvdata for picture data:'.format(rank))
        print(recvdata2)
        print("This is picture[:, i*n:i*n+n]")
        print(picture[:, i*n:i*n+n])

        if i == s-1:
            picture[:, (size_y-n-n_last):size_y] = recvdata2
        else:
            picture[:, i*n:i*n+n] = recvdata2

    plot_fun(picture, "foo1")

# comm.Barrier()

'''
