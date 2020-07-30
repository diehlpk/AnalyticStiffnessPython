# Reference solution for the 2d case
#@author Patrick Diehl (patrickdiehl@lsu.edu)
#@date June 2020
import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('font', size=14)


F=-4
E=4000
W=1.6
L=1.6
t=1
nu = 1/3

def ux(x):
    return F/(E*W*t)*(x-L)

def uy(y):
    return -((y/W)-0.5) / E / t * nu * F



h=0.1
n = int(1.6/h) + 1

nodes = []

for i in range(0,n):
    for j in range(0,n):
            nodes.append([i*h,j*h])

nodes = np.array(nodes)

# Plot u_x
plt.scatter(nodes[:,0],nodes[:,1],c=ux(nodes[:,0]))
ax = plt.gca()
ax.set_facecolor('#F0F8FF')
plt.xlabel("Positon $x$")
plt.ylabel("Poiston $y$")
v = np.linspace(min(abs(ux(nodes[:,0]))), max(abs(ux(nodes[:,0]))), 10, endpoint=True)
clb = plt.colorbar(ticks=v)
clb.set_label(r'Displacement $ u_x $',labelpad=5)
plt.savefig("ccm-2d-u-x.pdf",bbox_inches='tight')
plt.clf()
# Plot u_y
plt.scatter(nodes[:,0],nodes[:,1],c=uy(nodes[:,1]))
ax = plt.gca()
ax.set_facecolor('#F0F8FF')
plt.xlabel("Positon $x$")
plt.ylabel("Poiston $y$")
v = np.linspace(min(uy(nodes[:,1])), max(uy(nodes[:,1])), 10, endpoint=True)
clb = plt.colorbar(ticks=v)
clb.set_label(r'Displacement $ u_y $',labelpad=5)
plt.savefig("ccm-2d-u-y.pdf",bbox_inches='tight')



