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

    n = int(15/h) + 1


    nodes = []


    filehandler = open("bond-based-2d-plate-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-displacement.npy", "rb")
    uCurrent = np.load(filehandler, allow_pickle=True)

    filehandler = open("bond-based-2d-plate-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-damage.npy", "rb")
    damage = np.load(filehandler, allow_pickle=True)


    for i in range(0,n):
        for j in range(0,n):
            nodes.append([i*h,j*h])

    nodes = np.array(nodes)


    #plt.figure(figsize=(20,1))
    plt.scatter(nodes[:,0]+uCurrent[:,0],nodes[:,1]+uCurrent[:,1],c=uCurrent[:,1],cmap=grayscale_cmap(plt.get_cmap("viridis")),marker="s",s=np.sqrt(30))
    ax = plt.gca()
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top') 
    fig = plt.gcf()
    ax.set_facecolor('#F0F8FF')
    plt.xlabel(r"Position $x$")
    plt.ylabel(r"Poistion $y$")
    #plt.ylim([-2,21])
    #plt.xlim([0,max(nodes[:,1]+uCurrent[:,1])])
    v = np.linspace(min(uCurrent[:,1]), max(uCurrent[:,1]), 6, endpoint=True)
    clb = plt.colorbar(ticks=v,format='%.1e')
    clb.set_label(r'Displacement $u_y$')
    plt.savefig("bond-based-2d-plate-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-u-y-rotated.pdf",bbox_inches='tight')

    plt.clf()

    #plt.figure(figsize=(20,1))
    plt.scatter(nodes[:,0],nodes[:,1],c=damage,cmap=grayscale_cmap(plt.get_cmap("viridis")),marker="s",s=np.sqrt(30))
    ax = plt.gca()
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top') 
    fig = plt.gcf()
    ax.set_facecolor('#F0F8FF')
    plt.xlabel(r"Position $x$")
    plt.ylabel(r"Poistion $y$")
    #plt.ylim([-2,21])
    #plt.xlim([0,max(nodes[:,1])])
    v = np.linspace(min(damage), max(damage), 6, endpoint=True)
    clb = plt.colorbar(ticks=v,format='%.1e')
    clb.set_label(r'Damage')
    plt.savefig("bond-based-2d-plate-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-d-rotated.pdf",bbox_inches='tight')


  
