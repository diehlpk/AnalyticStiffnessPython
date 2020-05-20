#Protoype for the 1D bond-based analytic stiffness matrix
#@author Patrick Diehl
#@date May 2020
import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt

np.set_printoptions(precision=9)


class Compute:

    # Data
    nodes = []
    neighbors = []
    matrix = []
    uCurrent = []
    b = []
    f = []
    pertubation = []

    # Config
    h = 0.1
    delta = 4*h
    C = 0.000001
    beta = 1
    VB = h
    V = h
    eps = 1e-4

    def __init__(self):
        # Generate the mesh
        n = int(1/self.h)
        self.nodes = np.linspace(0,1,n) 
        # Search neighbors
        self.searchNeighbors()
        # Initialize 
        self.uCurrent = np.multiply(np.random.rand(len(self.nodes)),1e-5)
        self.b = np.zeros(len(self.nodes))
        self.b[len(self.nodes)-1] = 1. / self.h *0.0000001
        self.f = np.zeros(len(self.nodes))

    def searchNeighbors(self):
        for i in range(0,len(self.nodes)):
            self.neighbors.append([])
            for j in range(0,len(self.nodes)):
                if i != j and abs(self.nodes[j]-self.nodes[i]) <= self.delta:
                    self.neighbors[i].append(j)

    def L(self,i,j):
        nodes = self.nodes
        r = np.sqrt(self.length(i,j)) * self.S(i,j)
        return (2./self.VB) * self.w(self.length(nodes[i],nodes[j]))/self.delta  * self.f1(r) / np.sqrt(self.length(nodes[i],nodes[j])) *  self.S(i,j) *  self.e(i,j) * self.V


    def residual(self):
        for i in range(0,len(self.nodes)):
            self.f[i] = self.b[i]
            for j in self.neighbors[i]:
                self.f[i] += self.L(i,j)



    def length(self,x,y): 
        """ Computes the length between to nodes

        Keyword arguments:
        x -- the node x
        y -- the node y

        return the length
        """
        return np.sqrt((y-x) * (y-x)) 

 

    def S(self,i,j):
        """ Computes the stretch between node x and nod y

        Keyword arguments:
        x -- the node x
        y -- the node y
        u -- the displacement vector 

        return the stretch
        """
        nodes = self.nodes
        #print("S",i,j,np.dot((self.uCurrent[j] - self.uCurrent[i]) / self.length(nodes[i],nodes[j]),self.e(i,j)))
        return  np.dot((self.uCurrent[j] - self.uCurrent[i]) / self.length(nodes[i],nodes[j]),self.e(i,j))

    def w(self,r):
        """ Influence function

        Keyword arguments:
        r -- the initial length between node x and y

        return the weight
        """
        return 1

    def e(self,i,j):
        return (self.nodes[j]-self.nodes[i]) / self.length(self.nodes[i],self.nodes[j])

    def f1(self,r):
        return 2*r*self.C*self.beta* np.exp(-self.beta * r * r )

    def f2(self,r):
        return 2*self.C*self.beta*np.exp(-r*r*self.beta)-4*self.C*r*r*self.beta*self.beta*np.exp(-r*r*self.beta)

    def A(self,i,j):
        nodes = self.nodes
        r = np.sqrt(self.length(i,j)) * self.S(i,j)
        return  (2./self.VB) * self.w(self.length(nodes[i],nodes[j]))/self.delta  * self.f2(r)  * ( 1. / self.length(nodes[i],nodes[j]))  * self.E(i,j) * self.V


    def assemblymatrix(self):
        self.matrix = np.zeros((len(self.nodes),len(self.nodes)))
        for i in range(0,len(self.nodes)):
            for j in self.neighbors[i]:
                self.matrix[i][j] = self.A(i,j)
                self.matrix[i][i] -= self.A(i,j)


    def E(self,i,j):
        return np.dot(self.e(i,j),self.e(j,i))

    def solve(self,maxIt,epsilion):

        self.residual()
        print("Residual with intial guess",np.linalg.norm(self.f))
        it = 1
        residual = np.finfo(np.float).max

        while( residual > epsilion and it <= maxIt):

            self.assemblymatrix()
            self.matrix = np.delete(self.matrix,0,0)
            self.matrix = np.delete(self.matrix,0,1)
            res = linalg.solve(self.matrix, np.delete(self.f,0))
            self.uCurrent = np.concatenate([[0],res])  
            self.residual()
            residual = np.linalg.norm(self.f)
            print("Iteration ",it," Residual: ",residual)
            it += 1



    def plot(self):
        plt.plot(self.nodes,self.uCurrent)
        plt.xlabel("Positon")
        plt.ylabel("Displacement")
        plt.grid()
        plt.savefig("bond-based-1d.pdf")




if __name__=="__main__": 
    c = Compute()
    c.solve(10,1e-6)
    c.plot()