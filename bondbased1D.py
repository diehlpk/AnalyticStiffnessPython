#Protoype for the 1D bond-based analytic stiffness matrix
#@author Patrick Diehl
#@date May 2020
import numpy as np 
from scipy import linalg

np.set_printoptions(precision=1)


class Compute:

    # Data
    nodes = []
    neighbors = []
    matrix = []
    uCurrent = []
    uAct = []
    uInitial = []
    b = []
    f = []

    # Config
    h = 0.1
    delta = 3*h
    C = 1
    beta = 1
    VB = h

    def __init__(self):
        # Generate the mesh
        n = int(1/self.h)
        self.nodes = np.linspace(0,1,n) 
        # Search neighbors
        self.searchNeighbors()
        # Initialize 
        self.uCurrent = np.random.random_sample(size =(len(self.nodes))) 
        self.uAct = np.zeros(len(self.nodes))
        self.b = np.zeros(len(self.nodes))
        self.b[len(self.nodes)-1] = 10 / self.h
        self.b[0] = -10 / self.h
        self.f = np.zeros(len(self.nodes))
        self.residual()
        print("Residual with initial guess:",np.linalg.norm(self.f))



    def searchNeighbors(self):
        for i in range(0,len(self.nodes)):
            self.neighbors.append([])
            for j in range(0,len(self.nodes)):
                if i != j and abs(self.nodes[j]-self.nodes[i]) < self.delta:
                    self.neighbors[i].append(j)

    def L(self,i,j):
        nodes = self.nodes
        r = np.sqrt(self.length(i,j)) * self.S(i,j,self.uCurrent)
        return (2./self.VB) * self.w(self.length(nodes[i],nodes[j])/self.delta)  * self.f1(r) * ( 1. / self.length(nodes[i],nodes[j])) *  self.S(i,j,self.uCurrent) *  (nodes[j]-nodes[i]) / (self.length(nodes[i],nodes[j]))


    def residual(self):
        for i in range(0,len(self.nodes)):
            self.f[i] = -self.b[i]
            for j in self.neighbors[i]:
                self.f[i] -= self.L(i,j)



    def length(self,x,y): 
        """ Computes the length between to nodes

        Keyword arguments:
        x -- the node x
        y -- the node y

        return the length
        """
        return np.sqrt((y-x) * (y-x)) 

 

    def S(self,i,j,u):
        """ Computes the stretch between node x and nod y

        Keyword arguments:
        x -- the node x
        y -- the node y
        u -- the displacement vector 

        return the stretch
        """
        nodes = self.nodes
        return  (u[j] - u[i]) / self.length(nodes[i],nodes[j])  *  (nodes[j]-nodes[i]) / (self.length(nodes[i],nodes[j]))

    def w(self,r):
        """ Influence function

        Keyword arguments:
        r -- the initial length between node x and y

        return the weight
        """
        return 1

    def f1(self,r):
        return 2*r*self.C*self.beta* np.exp(-self.beta * r * r )

    def f2(self,r):
        return 2*self.C*self.beta*np.exp(-r*r*self.beta)-4*self.C*r*r*self.beta*self.beta*np.exp(-r*r*self.beta)

    def A(self,i,j):
        nodes = self.nodes
        r = np.sqrt(self.length(i,j)) * self.S(i,j,self.uCurrent)
        return (2./self.VB) * self.w(self.length(nodes[i],nodes[j])/self.delta)  * self.f2(r) * ( 1. / self.length(nodes[i],nodes[j])) *  self.S(i,j,self.uCurrent)  *  (nodes[j]-nodes[i]) / (self.length(nodes[i],nodes[j]))


    def assemblymatrix(self):
        self.matrix = np.zeros((len(self.nodes),len(self.nodes)))
        for i in range(0,len(self.nodes)):
            for j in self.neighbors[i]:
                self.matrix[i][j] =  self.A(i,j)
                self.matrix[i][i] -= self.A(i,j)

    def solve(self,maxIt,epsilion):


        for i in range(0,5):

            #print("uc",self.uCurrent)
            self.assemblymatrix()
            print(self.matrix)
            self.residual()
            self.uAct = linalg.solve(self.matrix, self.f)
            self.uCurrent = self.uAct
            print(self.f)
            print(np.linalg.norm(self.f))

        #print(self.uCurrent)
        #self.residual()
        #print(np.linalg.norm(self.f))
    

if __name__=="__main__": 
    c = Compute()
    c.solve(1,1)