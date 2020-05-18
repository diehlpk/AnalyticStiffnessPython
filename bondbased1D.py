#Protoype for the 1D bond-based analytic stiffness matrix
#@author Patrick Diehl
#@date May 2020
import numpy as np 
from scipy import linalg

nodes = []
neighbors = []
matrix = np.zeros((len(nodes),len(nodes)))

h = 0.1
delta = 3*h

C = 1
beta = 1

VB = h


def length(x,y): 
    """ Computes the length between to nodes

    Keyword arguments:
    x -- the node x
    y -- the node y

    return the length
    """
    return np.sqrt((y-x) * (y-x)) 

def S(i,j,u):
    """ Computes the stretch between node x and nod y

    Keyword arguments:
    x -- the node x
    y -- the node y
    u -- the displacement vector 

    return the stretch
    """
    return  (u[j] - u[i]) / length(nodes[i],nodes[j])  *  (nodes[j]-nodes[i]) / (length(nodes[i],nodes[j]))

def w(r):
    """ Influence function

    Keyword arguments:
    r -- the initial length between node x and y

    return the weight
    """
    return 1

def f1(r):
    return 2*r*C*beta* np.exp(-beta * r * r )

def f2(r):
    return 2*C*beta*np.exp(-r*r*beta)-4*C*r*r*beta*beta*np.exp(-r*r*beta)

def A(i,j,u):
    r = np.sqrt(length(i,j)) * S(i,j,u)
    return (2./VB) * w(length(nodes[i],nodes[j])/delta)  * f2(r) * ( 1. / length(nodes[i],nodes[j]))  *  (nodes[j]-nodes[i]) / (length(nodes[i],nodes[j]))

def L(i,j,u):
    r = np.sqrt(length(i,j)) * S(i,j,u)
    return (2./VB) * w(length(nodes[i],nodes[j])/delta)  * f1(r) * ( 1. / length(nodes[i],nodes[j]))  *  (nodes[j]-nodes[i]) / (length(nodes[i],nodes[j]))


def searchNeighbors():
    for i in range(0,len(nodes)):
        neighbors.append([])
        for j in range(0,len(nodes)):
            if i != j and abs(nodes[j]-nodes[i]) < delta:
                neighbors[i].append(j)


def assemblymatrix():
    matrix = np.zeros((10,10))
    for i in range(0,len(nodes)):
        for j in neighbors[i]:
            matrix[i][j] = A(i,j,u)
            matrix[i][i] -= A(i,j,u)
    return matrix

# Generate the mesh
n = int(1/h)
nodes = np.linspace(0,1,n)

# Search neighbors
searchNeighbors()

# Initialize 
u = np.zeros(len(nodes))
b = np.zeros(len(nodes))
b[0] = -10
b[len(nodes)-1] = 10


for i in range (0,3):
    # Assemble stiffness matrix
    matrix = assemblymatrix()
    f = np.zeros(len(nodes))
    # Generate the left-hand side
    for i in range(0,len(nodes)):
        f[i] = b[i]
        for j in neighbors[i]:
            f[i] += L(i,j,u)
    u = linalg.solve(matrix, f)
    print(i)



print(u)