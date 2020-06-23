#Protoype for the 2D bond-based analytic stiffness matrix
#@author Patrick Diehl (patrickdiehl@lsu.edu)
#@date June 2020
import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

np.set_printoptions(precision=4)
#mpl.style.use('seaborn')

class Compute:

    nodes = []
    neighbors = []

    # Config
    C = 4000
    beta = 100

    def __init__(self,h):
        # Generate the mesh
        self.h = h
        self.delta = 4*h
        n = int(16/self.h) + 1
        self.fix = []
        self.load = []
        #Generate grid
        index = 0
        for i in range(0,n):
            for j in range(0,n):
                self.nodes.append([i*h,j*h])
                if i*h < 3*h:
                    self.fix.append(index) 
                if i*h > 16-3*h:
                    self.load.append(index)
                index+=1
        
        self.fix = np.sort(self.fix)
        self.nodes = np.array(self.nodes)

        self.V = np.empty(n*n)
        self.V.fill(h*h)
        self.VB = np.pi * self.delta * self.delta
        # Search neighbors
        self.searchNeighbors()
        # Initialize 
        self.uCurrent = np.zeros(2*len(self.nodes)).reshape((len(self.nodes),2))
        
        #Apply the load to the body force vecotr
        self.b = np.zeros(2*len(self.nodes))
        for i in range(0,len(self.nodes)):
                if i in self.load:
                    self.b[2*i] = 40. / self.V[i]

        self.f = np.zeros(2*len(self.nodes))
     

    def searchNeighbors(self):
        for i in range(0,len(self.nodes)):
            self.neighbors.append([])
            for j in range(0,len(self.nodes)):
                if i != j and self.length(j,i) <= self.delta:
                    self.neighbors[i].append(j)

    def L(self,i,j):
        nodes = self.nodes
        r = np.sqrt(self.length(i,j)) * self.S(i,j)
        return (2./self.VB) * self.w(self.length(i,j))/self.delta  * self.f1(r) / self.length(i,j)  *  self.e(i,j) * self.V[j]


    def residual(self):
        self.f.fill(0)
        self.f += self.b

        for i in range(0,len(self.nodes)):
            for j in self.neighbors[i]:
                tmp =  self.L(i,j)
                self.f[2*i] += tmp[0]
                self.f[2*i+1] += tmp[1]
            if i in self.fix:
                self.f[2*i] = 0




    def length(self,i,j): 
        return np.sqrt((self.nodes[j][0]-self.nodes[i][0]) * (self.nodes[j][0]-self.nodes[i][0]) + (self.nodes[j][1]-self.nodes[i][1]) * (self.nodes[j][1]-self.nodes[i][1]) ) 

    def S(self,i,j):
        nodes = self.nodes
        return  np.dot((self.uCurrent[j] - self.uCurrent[i]) / self.length(i,j), self.e(i,j))

    def w(self,r):
        return 1

    def e(self,i,j):
        return (self.nodes[j]-self.nodes[i]) / self.length(i,j)

    def f1(self,r):
        return 2*r*self.C*self.beta* np.exp(-self.beta * r * r )

    def f2(self,r):
        return 2*self.C*self.beta*np.exp(-r*r*self.beta)-4*self.C*r*r*self.beta*self.beta*np.exp(-r*r*self.beta)

    def A(self,i,j):
        nodes = self.nodes
        r = np.sqrt(self.length(i,j)) * self.S(i,j)
        return  (2./self.VB) * self.w(self.length(i,j))/self.delta  * self.f2(r)  * ( 1. / self.length(i,j))  * self.E(i,j) 


    def assemblymatrix(self):
        self.matrix = np.zeros((2*len(self.nodes),2*len(self.nodes)))
        for i in range(0,len(self.nodes)):
            for j in self.neighbors[i]:
                tmp = self.A(i,j)
                #Set the matrix entries for the neighbors
                self.matrix[i*2][j*2] =  tmp[0,0]
                self.matrix[i*2][j*2+1] =  tmp[0,1]
                self.matrix[i*2+1][j*2] =  tmp[1,0]
                self.matrix[i*2+1][j*2+1] =  tmp[1,1]
                #set the matrix etnry for the node itself
                self.matrix[i*2][i*2] +=  tmp[0,0]
                self.matrix[i*2][i*2+1] +=  tmp[0,1]
                self.matrix[i*2+1][i*2] +=  tmp[1,0]
                self.matrix[i*2+1][i*2+1] +=  tmp[1,1]

        #plt.matshow(self.matrix)
        #plt.colorbar()
        #plt.show()


    def E(self,i,j):
        xi = self.e(i,j)
        return np.array([[xi[0]*xi[0], xi[0]*xi[1]],[xi[0]*xi[1],xi[1]*xi[1]]])
        
        
        #return np.tensordot(self.e(i,j),self.e(j,i),axes=0)

    def solve(self,maxIt,epsilion):

        self.residual()
        print("Residual with intial guess",np.linalg.norm(self.f))
        it = 1
        residual = np.finfo(np.float).max

        while( residual > epsilion):

            self.assemblymatrix()  

            b = np.copy(self.f)

            for i in range(len(self.fix)-1,0):
                
                b= np.delta(b,i)
                self.matrix = np.delete(self.matrix,i,0)
                self.matrix = np.delete(self.matrix,i,1)
    
            
            #length=2*len(self.nodes)-1
            #ixgrid = np.ix_([length, length-1], np.linspace(0,length,length+1,dtype=int))
            #print(self.nodes[len(self.nodes)-1])
            #print(self.matrix[ixgrid].tolist())
            #print(np.sum(np.absolute(self.matrix[ixgrid])))
            #print(np.nonzero(self.matrix[ixgrid][0]))
            #print(np.nonzero(self.matrix[ixgrid][1]))
            #print(len(self.neighbors[len(self.nodes)-1]))
            #print(self.b[2*len(self.nodes)-2])


            #length=2*len(self.nodes)-17
            #ixgrid = np.ix_([length, length-1], np.linspace(0,length,length+1,dtype=int))
            #print(self.matrix[ixgrid].tolist())
            #print(self.nodes[len(self.nodes)-17])
            #print(np.sum(np.absolute(self.matrix[ixgrid])))
            #print(np.nonzero(self.matrix[ixgrid][0]))
            #print(np.nonzero(self.matrix[ixgrid][1]))
            #print(len(self.neighbors[len(self.nodes)-17]))
            #print(self.b[2*len(self.nodes)-17-1])
            
            
      
            #sys.exit()
          



            res = linalg.solve(self.matrix,b)
    
            unew = np.zeros(2*len(self.nodes)).reshape((len(self.nodes),2))
            for i in range(0,len(self.uCurrent)):
                if not i in self.fix:
                    unew[i] = [res[2*i],res[2*i+1]]
                    
            self.uCurrent += unew

        
            #plt.scatter(self.nodes[:,0],self.nodes[:,1],c=res[0:len(self.f)-1:2])
            #plt.colorbar()
            #ax = plt.gca()
            #ax.set_facecolor('gray')
            #plt.show()

            self.residual()
            residual = np.linalg.norm(self.f) 
            print("Iteration ",it," Residual: ",residual)
            it += 1

        print(self.uCurrent[len(self.uCurrent)-1])

    def plot(self):
        # Plot u_x
        plt.scatter(self.nodes[:,0],self.nodes[:,1],c=self.uCurrent[:,0])
        plt.colorbar()
        ax = plt.gca()
        ax.set_facecolor('gray')
        plt.xlabel("Position")
        plt.ylabel("Displacement $u_x$")
        plt.savefig("bond-based-2d-u-x.pdf")
        # Plot u_y
        plt.scatter(self.nodes[:,0],self.nodes[:,1],c=self.uCurrent[:,1])
        plt.colorbar()
        ax = plt.gca()
        ax.set_facecolor('gray')
        plt.xlabel("Position")
        plt.ylabel("Displacement $u_y$")
        plt.savefig("bond-based-2d-u-y.pdf")


if __name__=="__main__": 

    c = Compute(1)
    c.solve(1000000,1e-11)
    #c.plot()