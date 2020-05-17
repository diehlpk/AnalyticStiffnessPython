import numpy as np 
from scipy import linalg

nodes = []

h = 0.1
delta = 3*h

C = 1
beta = 1

VB = h

def length(x,y): 
    return np.sqrt((y-x) * (y-x)) 

def S(i,j,u):
    return  (u[j] - u[i]) / length(nodes[i],nodes[j])  *  (nodes[j]-nodes[i]) / (length(nodes[i],nodes[j]))


def w(r):
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



# Generate the mesh

n = int(1/h)

nodes = np.linspace(0,1,n)

#print(nodes)

# Search neighbors

neighbors = []

for i in range(0,len(nodes)):
    neighbors.append([])
    for j in range(0,len(nodes)):
        if i != j and abs(nodes[j]-nodes[i]) < delta:
            neighbors[i].append(j)

#print(neighbors)


# Generate stiffness matrix

matrix = np.zeros((len(nodes),len(nodes)))

u = np.zeros(len(nodes))
b = np.zeros(len(nodes))
b[0] = -10
b[len(nodes)-1] = 10
f = np.zeros(len(nodes))


for i in range(0,len(nodes)):
    for j in neighbors[i]:
        matrix[i][j] = A(i,j,u)
        matrix[i][i] -= A(i,j,u)

# Generate the left-hand side

for i in range(0,len(nodes)):
    f[i] = b[i]
    for j in neighbors[i]:
        f[i] += L(i,j,u)


#print(matrix)
#print(f)

x = linalg.solve(matrix, f)



print(x)