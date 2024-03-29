#Protoype for the 1D bond-based analytic stiffness matrix
#@author Patrick Diehl (patrickdiehl@lsu.edu)
#@date May 2020
import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

np.set_printoptions(precision=4)
mpl.style.use('seaborn')

class Compute:

    # Data
    nodes = []
    neighbors = []
    matrix = []
    uCurrent = []
    b = []
    f = []

    # Config
    C = 4000
    beta = 1


    def __init__(self,h):
        # Generate the mesh
        self.h = h
        self.delta = 3*h
        n = int(16/self.h) + 1
        print(n)
        self.V = np.empty(n)
        self.V.fill(h)
        self.V[0] = h/2
        self.V[len(self.V)-1] = h/2

        self.VB = 3*self.delta
        self.nodes = np.linspace(0,16,n) 
        # Search neighbors
        self.searchNeighbors()
        # Initialize 
        self.uCurrent = np.zeros(len(self.nodes))
        #self.uCurrent = np.multiply(np.random.rand(len(self.nodes)),1e-2)
        self.b = np.zeros(len(self.nodes))
        self.b[len(self.nodes)-1] = 40. / self.V[len(self.nodes)-1]
        #self.b[len(self.nodes)-2] = 40. / self.V[len(self.nodes)-2] 
        #self.b[len(self.nodes)-3] = 40. / self.V[len(self.nodes)-3]  
        self.f = np.zeros(len(self.nodes))



    def searchNeighbors(self):
        """ Search all neighbors for each node"""

        for i in range(0,len(self.nodes)):
            self.neighbors.append([])
            for j in range(0,len(self.nodes)):
                if i != j and abs(self.nodes[j]-self.nodes[i]) <= self.delta:
                    self.neighbors[i].append(j)

    def L(self,i,j):
        """ Compute the peridynamic force 

        Keyword arguments
        i = Index of node i
        j = Index of node j
        """

        nodes = self.nodes
        r = np.sqrt(self.length(i,j)) * self.S(i,j)
        return (2./self.VB) * self.w(self.length(nodes[i],nodes[j]))/self.delta  * self.f1(r) / self.length(nodes[i],nodes[j])  *  self.e(i,j) * self.V[j]


    def residual(self):
        """ Compute the residual vector """

        for i in range(0,len(self.nodes)):
            self.f[i] = self.b[i]
            for j in self.neighbors[i]:
                self.f[i] += self.L(i,j)
        self.f[0] = 0

    def length(self,x,y): 
        """ Computes the distance between the node x and node y 

        Keyword arguments:
        x = node x
        y = node y
        """

        return np.sqrt((y-x) * (y-x)) 

    def S(self,i,j):
        """ Compute the strech between the nodes with the idnex i and index j

        Keyword arguments
        i = Index of node i
        j = Index of node j
        """

        nodes = self.nodes
        return  np.dot((self.uCurrent[j] - self.uCurrent[i]) / self.length(nodes[i],nodes[j]),self.e(i,j))

    def w(self,r):
        """ Influence function """

        return 1

    def e(self,i,j):
        """ Computes the normilazed distance between nodes with the index i and j in the intitial configuraiton """

        return (self.nodes[j]-self.nodes[i]) / self.length(self.nodes[i],self.nodes[j])

    def f1(self,r):
        """ Computes the potential for the peridyanmic force """

        return 2*r*self.C*self.beta* np.exp(-self.beta * r * r )

    def f2(self,r):
        """ Computes the potential for the approximation of the peridyanmic force """

        return 2*self.C*self.beta*np.exp(-r*r*self.beta)-4*self.C*r*r*self.beta*self.beta*np.exp(-r*r*self.beta)

    def A(self,i,j):
        """ Compute the tensor for the stifffness matrix for the nodes with the index i and index j

        Keyword arguments
        i = Index of node i
        j = Index of node j
        """

        nodes = self.nodes
        r = np.sqrt(self.length(i,j)) * self.S(i,j)
        return  (2./self.VB) * self.w(self.length(nodes[i],nodes[j]))/self.delta  * self.f2(r)  * ( 1. / self.length(nodes[i],nodes[j]))  * self.E(i,j) 


    def assemblymatrix(self):
        """ Assembles the stiffness matrix """

        self.matrix = np.zeros((len(self.nodes),len(self.nodes)))
        for i in range(0,len(self.nodes)):
            for j in self.neighbors[i]:
                self.matrix[i][j] = self.A(i,j)
                self.matrix[i][i] -= self.A(i,j)


    def E(self,i,j):
        """ Computes the distance tensor with respect to the intitial configuraiton """

        return np.dot(self.e(i,j),self.e(j,i))

    def solve(self,maxIt,epsilion):
        """ Runs the Newton step to determine the displacement solution """
        
        self.residual()
        print("Residual with intial guess",np.linalg.norm(self.f))
        it = 1
        residual = np.finfo(np.float).max

        while( residual > epsilion):

            self.assemblymatrix()
            b = np.copy(self.f)

            #Remove the column and row for the first fixed node
            self.matrix = np.delete(self.matrix,0,0)
            self.matrix = np.delete(self.matrix,0,1)
            
       
            b = np.delete(b,0)

            res = linalg.solve(self.matrix,b )
            self.uCurrent += np.concatenate([[0],res])  
            self.residual()
            residual = np.linalg.norm(self.f) 
            print("Iteration ",it," Residual: ",residual)
            it += 1

    def plot(self):
        plt.plot(self.nodes,self.uCurrent,color='#007FFF')
        plt.xlabel("Position")
        plt.ylabel("Displacement")
        plt.savefig("bond-based-1d.pdf")

if __name__=="__main__": 

    c = Compute(0.5)
    c.solve(1000000,1e-11)
    c.plot()
