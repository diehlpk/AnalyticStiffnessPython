import matplotlib.pyplot as plt
import numpy as np 
import pickle 
import sys
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

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
    max_i = d_small.index(max_d)
    print(nodes_small[max_i][0])
    d_pos = []
    d_index = []
   
    for i in range(0,len(d_small)):
        if d_small[i] <= 0.75 * max_d:
            d_small[i] = 0
        else:
            d_index.append(nodes_small[i][0])
            d_pos.append(nodes[i,0]+u_small[i][0])
   
    #print(max(d_pos))
    #print(d_pos)
    max_x = max(d_index)
    print(max_x)

    for i in range(0,len(d_small)):
        if nodes_small[i][0] < nodes_small[max_i][0]:
            d_small[i] = 0

    nodes_small = np.array(nodes_small)
    u_small = np.array(u_small)
    d_small = np.array(d_small)


    #plt.quiver(nodes_small[:,0],nodes_small[:,1],u_small[:,0],u_small[:,1])
    #plt.show()
            
          
    ax = plt.gca()
    plt.scatter(nodes_small[:,0]+u_small[:,0],nodes_small[:,1]+u_small[:,1],c=u_small[:,1],cmap=grayscale_cmap("viridis"),marker="s",s=np.sqrt(40)) 
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

    #cmap = mpl.cm.coolwarm
    #bounds = [-1, 2, 5, 7, 12, 15]
    #bounds = [0,2.1e-1,8.2e-1,1.9,5.0,6.4]
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')


    plt.scatter(nodes_small[:,0],nodes_small[:,1],c=d_small,cmap="coolwarm",marker="s",s=np.sqrt(30),vmin=0,vmax=6.4)
    ax = plt.gca()
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top') 
    fig = plt.gcf()
    ax.set_facecolor('#F0F8FF')
    plt.xlabel(r"Position $x$")
    plt.ylabel(r"Poistion $y$")
    #plt.ylim([-2,21])
    #plt.xlim([0,max(nodes[:,1])])
    #v = np.linspace(min(d_small), max(d_small), 6, endpoint=True)
    v =  np.linspace(0, 6.4, 6, endpoint=True)
    #v = [0,2.1e-1,8.2e-1,1,1.9,5.0,6.4]
    clb = plt.colorbar(ticks=v,format='%.1e')
    clb.set_label(r'Strain ratio w.r.t critical strain')
    #cmap = mpl.cm.coolwarm
    #norm = mpl.colors.Normalize(vmin=5, vmax=10)
    #fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    ax.hlines(y=7.5, xmin=0, xmax=nodes_small[max_i][0], linewidth=2, color='black')
    ax.vlines(x=nodes_small[max_i][0], ymin=7.2, ymax=7.8, color='white')
    plt.text(nodes_small[max_i][0], 8, r'crack tip', fontsize=8, color='white', horizontalalignment='center')
    ax.hlines(y=7, xmin=nodes_small[max_i][0], xmax=max_x, color='white')
    plt.text(0.5*(nodes_small[max_i][0]+max_x), 6.4, r'process zone', fontsize=8, color='white', horizontalalignment='center')
    plt.savefig("bond-based-2d-plate-"+str(h)+"-"+str(delta_factor)+"-"+str(iter)+"-d-rotated-hard.pdf",bbox_inches='tight')


  
