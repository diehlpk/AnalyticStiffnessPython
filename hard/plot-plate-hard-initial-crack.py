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


    filehandler = open("bond-based-2d-plate-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-displacement-hard.npy", "rb")
    uCurrent = np.load(filehandler, allow_pickle=True)

    filehandler = open("bond-based-2d-plate-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-damage-hard.npy", "rb")
    damage = np.load(filehandler, allow_pickle=True)

    for i in range(0,n):
        for j in range(0,n+2*delta_factor):
            nodes.append([i*h,(j-delta_factor)*h])
  
    nodes = np.array(nodes)

    #plt.figure(figsize=(20,1))

    nodes_small = []
    u_small = []
    d_small = []

    for i in range(0,len(nodes)):
        if nodes[i][1] >=0 and nodes[i][1] <= 15:
            
            nodes_small.append(nodes[i])
            u_small.append(uCurrent[i])
            d_small.append(damage[i])

    max_d = max(d_small)
    d_pos = []
   
    for i in range(0,len(d_small)):
        if d_small[i] <= 0.75 * max_d :
            d_small[i] = 0
        else:
            d_pos.append(nodes[i,0]+u_small[i][0])

    print(max(d_pos))
    print(d_pos)

    nodes_small = np.array(nodes_small)
    u_small = np.array(u_small)
    d_small = np.array(d_small)


    #plt.quiver(nodes_small[:,0],nodes_small[:,1],u_small[:,0],u_small[:,1])
    #plt.show()
            
          
    ax = plt.gca()
    plt.scatter(nodes_small[:,0]+u_small[:,0],nodes_small[:,1]+u_small[:,1],c=u_small[:,1],cmap=grayscale_cmap("viridis"),marker="s",s=np.sqrt(30)) 
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top') 
    fig = plt.gcf()
    ax.set_facecolor('#F0F8FF')
    plt.xlabel(r"Position $x$")
    plt.ylabel(r"Poistion $y$")
    #plt.ylim([-2,21])
    #plt.xlim([0,max(nodes[:,1]+uCurrent[:,1])])
    v = np.linspace(min(u_small[:,1]), max(u_small[:,1]), 6, endpoint=True)
    clb = plt.colorbar(ticks=v,format='%.1e')
    clb.set_label(r'Displacement $u_y$')
    #plt.show()
    plt.savefig("bond-based-2d-plate-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-u-y-rotated-hard.pdf",bbox_inches='tight')
    plt.clf()

    #plt.figure(figsize=(20,1)) grayscale_cmap(plt.get_cmap("viridis"))
    plt.scatter(nodes_small[:,0],nodes_small[:,1],c=d_small,cmap=grayscale_cmap("viridis"),marker="s",s=np.sqrt(30))
    ax = plt.gca()
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top') 
    fig = plt.gcf()
    ax.set_facecolor('#F0F8FF')
    plt.xlabel(r"Position $x$")
    plt.ylabel(r"Poistion $y$")
    #plt.ylim([-2,21])
    #plt.xlim([0,max(nodes[:,1])])
    v = np.linspace(min(d_small), max(d_small), 6, endpoint=True)
    clb = plt.colorbar(ticks=v,format='%.1e')
    clb.set_label(r'Damage')
    ax.hlines(y=7.5, xmin=0, xmax=7.5, linewidth=2, color='#FF7E00')
    plt.savefig("bond-based-2d-plate-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-d-rotated-hard.pdf",bbox_inches='tight')


  
