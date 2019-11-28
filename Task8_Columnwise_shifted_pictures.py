import numpy as np
import matplotlib.pyplot as plt
import imageio
import matplotlib.cm as cm
from mpi4py import MPI
import time

name = 'animation_walking.gif'
name_pict = "foo"

start_time = time.time()


def plot_fun(set, name_pict):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(set, cmap=cm.gray)
    plt.savefig("fig/"+name_pict+".png")


def anim():
    with imageio.get_writer(name, duration=0.5) as writer:
        for i in range(0, step):
         writer.append_data(imageio.imread("fig/"+name_pict+str(i)+".png"))


step = 11

comm = MPI.COMM_WORLD
size_x, size_y = 10, 10
coeff = 0.9
s = comm.size

n = size_y//s
n_last = size_y - n*s
rank = comm.rank

comm.barrier()

if comm.rank != s-1:

    part_picture = np.random.choice([0, 1], (size_x, n+2), replace=True, p=[coeff, 1.0 - coeff])
    print('DOING rand_picture')
    if rank != 0:
        comm.send(part_picture, dest=0)#, tag=comm.rank)
        print('Process {} sent data:'.format(rank), part_picture)

else:
    part_picture = np.random.choice([0, 1], (size_x, n+n_last+2), replace=True, p=[coeff, 1.0 - coeff])
    print('DOING rand_picture')

    comm.send(part_picture, dest=0)#, tag=comm.rank)
    print('Process {} sent data:'.format(rank), part_picture)

comm.barrier()

if comm.rank == 0:
    picture = np.zeros((size_x, size_y))
    picture[:, 0:n] = part_picture[:, 1: -1]

    for i in range(1, s):
        recvdata = comm.recv(source=i)#, tag=i)
        # recvdata = request.wait()
        print('Process {} recvdata[:, 1:-1]] data:'.format(rank))
        print(recvdata[:, 1:-1])
        print("This is picture[:, i*n:i*n+n]")
        print(picture[:, i*n:i*n+n])

        if i == s-1:
            picture[:, (size_y-n-n_last):size_y] = recvdata[:, 1:-1]
        else:
            picture[:, i*n:i*n+n] = recvdata[:, 1:-1]

    print("INIT PICTURE")
    print(picture)

    plot_fun(picture, "foo0")

comm.barrier()

# loop

for j in range(1, step):

    if comm.rank != s-1:
        for i in range(size_x):
            part_picture[i] = np.roll(part_picture[i], 1)
        print('ROLL part_picture')

        comm.send(part_picture[:, -1], dest=comm.rank+1)#, tag=comm.rank)
        print('Process {} sent ROLL data:'.format(rank), part_picture[:, -1])

    else:
        for i in range(size_x):
            part_picture[i] = np.roll(part_picture[i], 1)
        print('ROLL part_picture')

        comm.send(part_picture[:, -1], dest=0)#, tag=comm.rank)
        print('Process {} sent ROLL data:'.format(rank), part_picture[:, -1])

    comm.barrier()

    # combine the picture

    if comm.rank == 0:
        recvdata1 = comm.recv(source=s-1)#, tag=s-1)
        print('Process {} recvdata1 :'.format(rank))
        part_picture[:, 1] = recvdata1 #.reshape()

    else:
        recvdata1 = comm.recv(source=comm.rank-1)#, tag=comm.rank-1)
        print('Process {} recvdata1 :'.format(rank))
        part_picture[:, 1] = recvdata1

        comm.send(part_picture[:, 1: -1], dest=0)#, tag=comm.rank+s)
        print('Process {} sent ROLL data for picture:'.format(rank), part_picture[:, 1: -1])


    comm.barrier()

    if comm.rank == 0:
        picture = np.zeros((size_x, size_y))
        picture[:, 0:n] = part_picture[:, 1: -1]

        for i in range(1, s):
            recvdata2 = comm.recv(source=i)#, tag=i+s)
            # recvdata2 = request2.wait()
            print('Process {} recvdata for picture data:'.format(rank))
            print(recvdata2)
            print("This is picture[:, i*n:i*n+n]")
            print(picture[:, i*n:i*n+n])

            if i == s-1:
                picture[:, (size_y-n-n_last):size_y] = recvdata2
            else:
                picture[:, i*n:i*n+n] = recvdata2

        plot_fun(picture, "foo"+ str(j))

comm.barrier()

if comm.rank == 0:
    anim()
    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
