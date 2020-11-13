import matplotlib.pyplot as plt
import numpy as np 
import pickle 
import sys
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('font', size=17)


def grayscale_cmap(cmap):
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
        
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    


if __name__=="__main__": 


    h = float(sys.argv[1])
    delta_factor = int(sys.argv[2])
    iter = int(sys.argv[3])

    ny = int(100/h) + 1
    ny2 = int((50-h)/h) + 1
    nx = int(12/h) + 1
    nx2 = int(19/h) + 1

    nodes = []


    filehandler = open("bond-based-2d-tensile-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-displacement.npy", "rb")
    uCurrent = np.load(filehandler, allow_pickle=True)

    filehandler = open("bond-based-2d-tensile-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-damage.npy", "rb")
    damage = np.load(filehandler, allow_pickle=True)


    for i in range(0,nx2):
        for j in range(0,ny2):
            nodes.append([i*h,j*h])

    for i in range(0,nx):
        for j in range(0,ny):
            nodes.append([3.5+i*h,50+j*h])
        
    for i in range(0,nx2):
        for j in range(0,ny2):
            nodes.append([i*h,150+h+j*h])

    nodes = np.array(nodes)


    plt.figure(figsize=(20,1))
    plt.scatter(nodes[:,1]+uCurrent[:,1],nodes[:,0]+uCurrent[:,0],c=uCurrent[:,1],cmap=grayscale_cmap(plt.get_cmap("viridis")))
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    fig = plt.gcf()
    ax.set_facecolor('#F0F8FF')
    plt.xlabel(r"Position $y$")
    plt.ylabel(r"Poistion $x$")
    plt.ylim([-2,21])
    plt.xlim([0,max(nodes[:,1]+uCurrent[:,1])])
    v = np.linspace(int(min(uCurrent[:,1])), int(max(uCurrent[:,1])), 3, endpoint=True)
    clb = plt.colorbar(ticks=v,orientation="horizontal")
    clb.set_label(r'Displacement $u_y$')
    plt.savefig("bond-based-2d-tensile-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-u-y-rotated.pdf",bbox_inches='tight')

    plt.clf()

    plt.figure(figsize=(20,1))
    plt.scatter(nodes[:,1],nodes[:,0],c=damage,cmap=grayscale_cmap(plt.get_cmap("viridis")))
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    fig = plt.gcf()
    ax.set_facecolor('#F0F8FF')
    plt.xlabel(r"Position $y$")
    plt.ylabel(r"Poistion $x$")
    plt.ylim([-2,21])
    plt.xlim([0,max(nodes[:,1])])
    v = np.linspace(min(damage), max(damage), 3, endpoint=True)
    clb = plt.colorbar(ticks=v,orientation="horizontal")
    clb.set_label(r'Damage')
    plt.savefig("bond-based-2d-tensile-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-d-rotated.pdf",bbox_inches='tight')


  