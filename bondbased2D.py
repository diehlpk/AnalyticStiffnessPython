#Protoype for the 2D bond-based analytic stiffness matrix
#@author Patrick Diehl (patrickdiehl@lsu.edu)
#@date June 2020
import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('font', size=14)

np.set_printoptions(precision=4)
#mpl.style.use('seaborn')

class Compute:

    nodes = []
    neighbors = []

    # Config
    E=4000
    G=E/(2*(1+1/3))
    #beta=(16/6)*1e3/E
    beta=1
    C=30000.0
    #C=20*G/beta
    

    def __init__(self,h,delta_factor):
        # Generate the mesh
        self.h = h
        self.delta_factor = delta_factor
        self.delta = self.delta_factor*h
        n = int(1.6/self.h) + 1
        self.fix = []
        self.load = []
        #Generate grid
        index = 0
        for i in range(0,n):
            for j in range(0,n):
                self.nodes.append([i*h,j*h])
                if i*h < 1*h:
                    self.load.append(index)
                    #plt.scatter(i*h,j*h)
                if i*h > 1.6-1*h:
                    self.fix.append(index)
                    #plt.scatter(i*h,j*h)
                index+=1
        #plt.show()
        
        self.fix = np.sort(self.fix)

        self.nodes = np.array(self.nodes)

        self.V = np.empty(n*n)
        #self.V.fill(h*h)
        #self.V[0] *= 0.5
        #self.V[1] *= 0.5
        #self.V[2] *= 0.5
        #self.V[3] *= 0.5
        #self.V[4] *= 0.5
        #self.V[5] *= 0.5
        #self.V[6] *= 0.5
        #self.V[7] *= 0.5
        #self.V[8] *= 0.5
        #self.V[9] *= 0.5
        #self.V[10] *= 0.5
        #self.V[11] *= 0.5
        #self.V[12] *= 0.5
        #self.V[13] *= 0.5
        #self.V[14] *= 0.5
        #self.V[15] *= 0.5
        #self.V[16] *= 0.5

        #last = len(self.V) -1
        #self.V[last] *= 0.5
        #self.V[last-1] *= 0.5
        #self.V[last-2] *= 0.5
        #self.V[last-3] *= 0.5
        #self.V[last-4] *= 0.5
        #self.V[last-5] *= 0.5
        #self.V[last-6] *= 0.5
        #self.V[last-7] *= 0.5
        #self.V[last-8] *= 0.5
        #self.V[last-9] *= 0.5
        #self.V[last-10] *= 0.5
        #self.V[last-11] *= 0.5
        #self.V[last-12] *= 0.5
        #self.V[last-13] *= 0.5
        #self.V[last-14] *= 0.5
        #self.V[last-15] *= 0.5
        #self.V[last-16] *= 0.5

        self.V.fill(h*h)

        #for i in range(0,len(self.nodes)):
        #    plt.scatter(self.nodes[:,0],self.nodes[:,1],c=self.V)
        #plt.show()

        self.VB = np.pi * self.delta * self.delta
        # Search neighbors
        self.searchNeighbors()
        # Initialize 
        self.uCurrent = np.zeros(2*len(self.nodes)).reshape((len(self.nodes),2))
        
        #Apply the load to the body force vector
        self.b = np.zeros(2*len(self.nodes))
        for i in range(0,len(self.nodes)):
                if i in self.load:
                    self.b[2*i] = -0.4 

        self.f = np.zeros(2*len(self.nodes))

        print("Matrix size "+str(2*len(self.nodes))+"x"+str(2*len(self.nodes)))
     

    def searchNeighbors(self):
        for i in range(0,len(self.nodes)):
            self.neighbors.append([])
            for j in range(0,len(self.nodes)):
                if i != j and self.length(j,i) <= self.delta:
                    self.neighbors[i].append(j)

    def L(self,i,j):
        r = np.sqrt(self.length(i,j)) * self.S(i,j)
        return (2./self.VB) * self.w(self.length(i,j))/self.delta  * self.f1(r) / self.length(i,j)  *  self.e(i,j) * self.V[j]


    def residual(self):
        self.f.fill(0)
        self.f += self.b

        for i in range(0,len(self.nodes)):
            for j in self.neighbors[i]:
                tmp =  self.L(i,j)
                self.f[2*i] += tmp[0] * self.V[i]
                self.f[2*i+1] += tmp[1] * self.V[i]
        
            if i in self.fix:
                self.f[2*i] = 0
                self.f[2*i+1] = 0

    def length(self,i,j): 
        return np.sqrt((self.nodes[j][0]-self.nodes[i][0]) * (self.nodes[j][0]-self.nodes[i][0]) + (self.nodes[j][1]-self.nodes[i][1]) * (self.nodes[j][1]-self.nodes[i][1]) ) 

    def S(self,i,j):
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
        r = np.sqrt(self.length(i,j)) * self.S(i,j)
        return  (2./self.VB) * self.w(self.length(i,j))/self.delta  * self.f2(r)  * ( 1. / self.length(i,j))  * self.E(i,j) 


    def assemblymatrix(self):
        self.matrix = np.zeros((2*len(self.nodes),2*len(self.nodes)))
        for i in range(0,len(self.nodes)):
            if not i in self.fix:
                for j in self.neighbors[i]:
                    tmp = self.A(i,j)
                    #Set the matrix entries for the neighbors
                    self.matrix[i*2][j*2] +=  tmp[0,0]
                    self.matrix[i*2][j*2+1] +=  tmp[0,1]
                    self.matrix[i*2+1][j*2] +=  tmp[1,0]
                    self.matrix[i*2+1][j*2+1] +=  tmp[1,1]
                    #set the matrix entries for the node it self
                    self.matrix[i*2][i*2] -=  tmp[0,0]
                    self.matrix[i*2][i*2+1] -=  tmp[0,1]
                    self.matrix[i*2+1][i*2] -=  tmp[1,0]
                    self.matrix[i*2+1][i*2+1] -=  tmp[1,1]

    def E(self,i,j):
        #xi = self.e(i,j)
        #xj = self.e(j,i)
        #return np.array([[xi[0]*xi[0], xi[0]*xi[1]],[xi[0]*xi[1],xi[1]*xi[1]]])
        return np.tensordot(self.e(i,j),self.e(j,i),axes=0)

    def solve(self,maxIt,epsilion):

        self.residual()
        print("Residual with intial guess",np.linalg.norm(self.f))
        it = 1
        residual = np.finfo(np.float).max

        while(residual > epsilion):

            self.assemblymatrix()  

            b = np.copy(self.f)
            
            for i in range(0,len(self.fix)):

                index = 2* self.fix[len(self.fix)-1-i]
                b = np.delete(b,index+1)
                b = np.delete(b,index)
                self.matrix = np.delete(self.matrix,index+1,0)
                self.matrix = np.delete(self.matrix,index+1,1)   
                self.matrix = np.delete(self.matrix,index,0)
                self.matrix = np.delete(self.matrix,index,1)

            
            res = linalg.solve(self.matrix,b)
    
            unew = np.zeros(2*len(self.nodes)).reshape((len(self.nodes),2))
            j = 0
            for i in range(0,len(self.uCurrent)):
                if not i in self.fix: 
                    unew[i] = [res[2*j],res[2*j+1]]
                    j += 1
                    
            self.uCurrent += unew

            self.residual()
            residual = np.linalg.norm(self.f) 
            print("Iteration ",it," Residual: ",residual)
            it += 1

    def ux(self,x):
        F=-0.4
        E=4000
        W = L = 16
        t = 1
        return F/(E*W*t)*(x-L)

    def uy(self,y):
        F=-0.4
        E=4000
        W = L = 16
        t = 1
        nu = 1/3
        return -(y/W-0.5) / E / t * nu * F

    def plot(self):
        # Plot u_x
        plt.scatter(self.nodes[:,0],self.nodes[:,1],c=abs(self.uCurrent[:,0]))
        ax = plt.gca()
        ax.set_facecolor('#F0F8FF')
        v = np.linspace(min(abs(self.uCurrent[:,0])), max(abs(self.uCurrent[:,0])), 10, endpoint=True)
        clb = plt.colorbar(ticks=v)
        clb.set_label(r'Displacement $ u_x $',labelpad=5)
        plt.xlabel("Position $x$")
        plt.ylabel("Position $y$")
        plt.savefig("bond-based-2d-u-x-"+str(self.h)+"-"+str(self.delta_factor)+".pdf",bbox_inches='tight')
        plt.clf()
        # Plot u_y
        plt.scatter(self.nodes[:,0],self.nodes[:,1],c=self.uCurrent[:,1])
        ax = plt.gca()
        ax.set_facecolor('#F0F8FF')
        v = np.linspace(min(self.uCurrent[:,1]), max(self.uCurrent[:,1]), 10, endpoint=True)
        clb = plt.colorbar(ticks=v)
        clb.set_label(r'Displacement $ u_y $',labelpad=5)
        plt.xlabel("Position $x$")
        plt.ylabel("Position $y$")
        plt.savefig("bond-based-2d-u-y-"+str(self.h)+"-"+str(self.delta_factor)+".pdf",bbox_inches='tight')
        plt.clf()
        # Error plots
        eux = abs(abs(self.uCurrent[:,0])-abs(self.ux(self.nodes[:,0])))
        plt.scatter(self.nodes[:,0],self.nodes[:,1],c=eux)
        ax = plt.gca()
        ax.set_facecolor('#F0F8FF')
        v = np.linspace(min(eux), max(eux), 10, endpoint=True)
        clb = plt.colorbar(ticks=v)
        clb.set_label(r'Error $ u_x $',labelpad=5)
        plt.xlabel("Position $x$")
        plt.ylabel("Position $y$")
        plt.savefig("bond-based-2d-e-x-"+str(self.h)+"-"+str(self.delta_factor)+".pdf",bbox_inches='tight')
        plt.clf()
        # Plot e_y
        euy = abs(abs(self.uCurrent[:,1])-abs(self.uy(self.nodes[:,1])))
        plt.scatter(self.nodes[:,0],self.nodes[:,1],c=euy)
        ax = plt.gca()
        ax.set_facecolor('#F0F8FF')
        v = np.linspace(min(euy), max(euy), 10, endpoint=True)
        clb = plt.colorbar(ticks=v)
        clb.set_label(r'Error $ u_y $',labelpad=5)
        plt.xlabel("Position $x$")
        plt.ylabel("Position $y$")
        plt.savefig("bond-based-2d-e-y-"+str(self.h)+"-"+str(self.delta_factor)+".pdf",bbox_inches='tight')
        plt.clf()
        # Force plots
        tmp = self.f.reshape((len(self.nodes),2)) 
        plt.scatter(self.nodes[:,0],self.nodes[:,1],c=tmp[:,0])
        ax = plt.gca()
        ax.set_facecolor('#F0F8FF')
        v = np.linspace(min(tmp[:,0]), max(tmp[:,0]), 10, endpoint=True)
        clb = plt.colorbar(ticks=v)
        clb.set_label(r'$ f_x $',labelpad=5)
        plt.xlabel("Position $x$")
        plt.ylabel("Position $y$")
        plt.savefig("bond-based-2d-f-x-"+str(self.h)+"-"+str(self.delta_factor)+".pdf",bbox_inches='tight')
        plt.clf()
        # Plot f_y
        plt.scatter(self.nodes[:,0],self.nodes[:,1],c=tmp[:,1])
        ax = plt.gca()
        ax.set_facecolor('#F0F8FF')
        v = np.linspace(min(tmp[:,1]), max(tmp[:,1]), 10, endpoint=True)
        clb = plt.colorbar(ticks=v)
        clb.set_label(r'$ f_y $',labelpad=5)
        plt.xlabel("Position $x$")
        plt.ylabel("Position $y$")
        plt.savefig("bond-based-2d-f-y-"+str(self.h)+"-"+str(self.delta_factor)+".pdf",bbox_inches='tight')
        plt.clf()

if __name__=="__main__": 

    c = Compute(float(sys.argv[1]),int(sys.argv[2]))
    c.solve(1000000,1e-6)
    c.plot()