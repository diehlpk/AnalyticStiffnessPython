#Protoype for the 2D bond-based analytic stiffness matrix including fracture
# using a pre-crack square plate (hard loading)
#@author Patrick Diehl (patrickdiehl@lsu.edu)
#@date September 2020
import numpy as np 
from scipy import linalg
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import pickle 
import random
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
    beta=10
    C=300000000
    rbar=np.sqrt(0.5/beta)
 
    def __init__(self,h,delta_factor,iter):
        # Generate the mesh
        self.h = h
        self.delta_factor = delta_factor
        self.delta = self.delta_factor*h
        n = int(15/self.h) + 1
        self.fix = []
        self.loadT = []
        self.loadB = []
        self.z = []
        self.iter = iter

        #Generate grid
        #index = 0
        for i in range(0,n):
            for j in range(0,n+2*self.delta_factor):
                self.nodes.append([i*h,(j-self.delta_factor)*h])            
                #index += 1    

        self.nodes = np.array(self.nodes)

        #plt.show()
        #sys.exit(1)
 
        #self.fix = np.sort(self.fix)

        #self.nodes = np.array(self.nodes)

        self.V = np.empty(len(self.nodes))
        self.V.fill(h*h)

        self.d = np.zeros(len(self.nodes))

        self.VB = np.pi * self.delta * self.delta
        # Search neighbors
        self.searchNeighbors()

        self.f = np.zeros(2*len(self.nodes))

        self.b = np.zeros(2*len(self.nodes))

        if self.iter == 0:

            # Initialize 
            self.uCurrent = np.zeros(2*len(self.nodes)).reshape((len(self.nodes),2))
            self.wCurrentExtension = np.zeros(2*len(self.nodes)).reshape((len(self.nodes),2))
            self.wCurrentDomain = np.zeros(2*len(self.nodes)).reshape((len(self.nodes),2))

            #Apply the boundary load to the extension

            #print(min(self.nodes[:,1]))
            #scale = 1/7

            for i in range(0,len(self.nodes)):
                #if self.nodes[i][1] <= 0  : #and self.nodes[i][0] < 2.4  :
                #    self.wCurrentExtension[i][1] = (.125  / 11.5)  * (11.5 -self.nodes[i][1]) 
                #    self.loadB.append(i)
                #   
                #elif self.nodes[i][1] >= 15 : #and self.nodes[i][0] < 2.4 :
                #    self.wCurrentExtension[i][1] = 0 #-0.125 
                #    self.loadT.append(i)    
                #
                #elif self.nodes[i][1] > 0 and self.nodes[i][1] <= 2.5  :
                #    self.wCurrentDomain[i][1] = (.125  / 11.5)  * (11.5 -self.nodes[i][1]) 
                #
                #elif self.nodes[i][1] < 15 and self.nodes[i][1] >= 2.5  :
                #    self.wCurrentDomain[i][1] = 0 # (-.125  / 7.5)  * (self.nodes[i][1]-7.5) 

            
                if self.nodes[i][1] > 7.5 :
                    self.wCurrentExtension[i][1] = (-1e-1 ) * (self.nodes[i][1] - 7.5 ) / 15.5  - 1e-2  #  * (15 - self.nodes[i][0] )  / 15
                    self.loadB.append(i)

                if self.nodes[i][1] < 7.5 :
                    self.wCurrentExtension[i][1] = (1e-1 ) * ( -7.5 + self.nodes[i][1]  )  / -15.5  + 1e-2 # * (15 - self.nodes[i][0] )  / 15
                    self.loadT.append(i) 

                #if self.nodes[i][1] >= 15 :
                #    self.wCurrentExtension[i][1] = -1e-4

                #if self.nodes[i][1] <= 0 :
                #    self.wCurrentExtension[i][1] = 1e-4
                

                #if self.nodes[i][0] > 15 - h and self.nodes[i][1] >= 7.5 and self.nodes[i][1] < 7.5 + 1*h  :
                #    self.fix.append(i)  
                #    self.wCurrentExtension[i][1] = 0
                #    print("d")

                #elif self.nodes[i][0] > 15-h  and self.nodes[i][1] < 7.5 and self.nodes[i][1] >= 7.5 - 1*h  :
                #   self.fix.append(i)
                #    self.wCurrentExtension[i][1] = 0
                #    print("d2")

                if self.nodes[i][0] > 15 - h and self.nodes[i][1] > 15  + self.delta - h :
                    self.fix.append(i)  
                    self.wCurrentExtension[i][1] = 0
                    print("d")

                elif self.nodes[i][0] > 15-h  and self.nodes[i][1] < 7.5 and self.nodes[i][1] < 0 - self.delta + h :
                    self.fix.append(i)
                    self.wCurrentExtension[i][1] = 0
                    print("d2")
                

                #elif self.nodes[i][1] >= 15 and self.nodes[i][0] > 2.4 and  self.nodes[i][0] <= 15  :
                 #   self.wCurrent[i][1] = -.125 / 5  
                
                #elif self.nodes[i][1] <= 0  and  self.nodes[i][0] > 2.4 and  self.nodes[i][0] <= 15   :
                 #   self.wCurrent[i][1] = .125 / 5

                #else:
                    #plt.scatter(self.nodes[i][0],self.nodes[i][1],color="red")

            self.fix = np.sort(self.fix)
            print(self.fix)
     
            #self.remove = np.concatenate([self.fix,self.loadT,self.loadB])
            #self.remove = np.sort(self.remove)
                
                    
            #plt.scatter(self.nodes[:,0],self.nodes[:,1],c= self.wCurrentExtension[:,1],cmap="seismic")
            #for i in self.fix :
            #    plt.scatter(self.nodes[i,0],self.nodes[i,1],c="black")
            #plt.colorbar()
            #plt.show()
            #sys.exit(1)

        else:
            print("Restart not implemented yet")
            #filehandler = open("bond-based-2d-plate-"+str(self.h)+"-"+str(self.delta_factor)+"-"+str(self.iter-1)+"-displacement.npy", "rb")
            #self.nodes += np.load(filehandler)
            #self.uCurrent = np.load(filehandler)

        print("Matrix size "+str(2*len(self.nodes))+"x"+str(2*len(self.nodes)))
     

    def searchNeighbors(self):
        left = np.array([0,7.5])
        right = np.array([7.5,7.5])
        neighbor = []


        #fig = plt.gcf()
        #fig.set_size_inches(4,50)

        for i in range(0,len(self.nodes)):
            self.neighbors.append([])
            for j in range(0,len(self.nodes)):
                if i != j and self.length(j,i) <= self.delta:
                    if not self.intersect(left,right,self.nodes[i],self.nodes[j]):
                        self.neighbors[i].append(j)
                        #plt.plot([self.nodes[i][0],self.nodes[j][0]],[self.nodes[i][1],self.nodes[j][1]],c="#91A3B0",alpha=0.15)
            #neighbor.append(len(self.neighbors[i]))

        #plt.scatter(self.nodes[:,0],self.nodes[:,1],c=neighbor)
        #plt.xlabel("Position $x$",fontsize = 30)
        #plt.ylabel("Position $y$",fontsize = 30)
        #clb = plt.colorbar()
        #clb.set_label(r'$B_\delta(x)$',labelpad=5)
        #plt.show()
        #sys.exit()

    def L(self,i,j):
        r = np.sqrt(self.length(i,j)) * self.S(i,j)
        return (2./self.VB) * self.w(self.length(i,j))/self.delta  * self.f1(r) / self.length(i,j)  *  self.e(i,j) * self.V[j]

    def computeLoad(self,iter):
        self.b.fill(0)

        #uCurrentCopy = np.copy(self.uCurrent)

        self.uCurrent += iter * self.wCurrentExtension

        #if iter == 1:
        #self.uCurrent += iter * self.wCurrentDomain

        for i in range(0,len(self.nodes)):
            for j in self.neighbors[i]:
                
                tmp =  self.L(i,j)
                self.b[2*i] += tmp[0] 
                self.b[2*i+1] += tmp[1] 

        self.uCurrent -= iter * self.wCurrentExtension

        #if iter == 1:
        #self.uCurrent -= iter * self.wCurrentDomain

    def residual(self,iter):
        self.f.fill(0)

        #self.uCurrent =+ iter * self.wCurrent

        for i in range(0,len(self.nodes)):
            for j in self.neighbors[i]:
                
                tmp =  self.L(i,j)
                self.f[2*i] += tmp[0] 
                self.f[2*i+1] += tmp[1] 

        #self.uCurrent =- iter * self.wCurrent


    def computeDamage(self):

        for i in range(0,len(self.nodes)):
            for j in self.neighbors[i]:
                
                tmp =  self.L(i,j)
                self.f[2*i] += tmp[0] 
                self.f[2*i+1] += tmp[1] 

                self.damage(i,j)

    def damage(self,i,j):
        z = self.d[i]
        sr = 0
        sij = self.S(i,j) 
        rij = self.length(i,j)

        if rij > 1e-12:
            sr = np.abs(sij) / (self.rbar / np.sqrt(rij) )
        if z < sr :
            z = sr

        self.d[i] = z

        
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
        self.matrix = np.zeros((2*len(self.nodes),2*len(self.nodes)),dtype=np.double)

        
        for i in range(0,len(self.nodes)):
            
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
        return np.tensordot(self.e(i,j),self.e(j,i),axes=0)

    def intersect(self,A,B,C,D):
        line = LineString([(A[0], A[1]), (B[0], B[1])])
        other = LineString([(C[0], C[1]), (D[0], D[1])])
        return line.intersects(other)

    def solve(self,maxIt,epsilion):

        for iter in range(1,maxIt+1):

        
            print(" ##### Load step: " + str(iter+self.iter) + " #####")
            #self.residual(iter)

            #print("Residual with intial guess",np.linalg.norm(self.f))


            residual = np.finfo(float).max
            residual_old = 0

            self.computeLoad(iter)

            if iter < 0 :
            
                self.assemblymatrix()
                for i in range(0,len(self.fix)):
                    index = int(2* self.fix[len(self.fix)-1-i])
                    self.matrix = np.delete(self.matrix,index+1,1)
                    self.matrix = np.delete(self.matrix,index+1,0)   
                    self.matrix = np.delete(self.matrix,index,1)
                    self.matrix = np.delete(self.matrix,index,0)
                    
            it = 1
            while(residual > epsilion):

                
                if iter >=  1:
                    self.assemblymatrix()  
                    
                    b = np.copy(self.b)
            
                    for i in range(0,len(self.fix)):
                
                        index = int(2* self.fix[len(self.fix)-1-i])
                        b = np.delete(b,index+1)
                        b = np.delete(b,index)
                        self.matrix = np.delete(self.matrix,index+1,1)
                        self.matrix = np.delete(self.matrix,index+1,0)   
                        self.matrix = np.delete(self.matrix,index,1)
                        self.matrix = np.delete(self.matrix,index,0)

                else:

                    b = np.copy(self.b)
                    for i in range(0,len(self.fix)):
                        index = int(2* self.fix[len(self.fix)-1-i])
                        b = np.delete(b,index+1)
                        b = np.delete(b,index)
                

                inv = linalg.inv(self.matrix)

                res = inv.dot(b)
        
                unew = np.zeros(2*len(self.nodes)).reshape((len(self.nodes),2))
                j = 0
                for i in range(0,len(self.uCurrent)):
                    if not i in self.fix: 
                        unew[i] = np.array([res[2*j],res[2*j+1]])
                        j += 1
                    #elif i in self.loadT:
                    #    unew[i] = iter* np.array([0,0.125])
                    #elif i in self.loadB:
                    #    unew[i] = iter * np.array([0,-.125])

                #plt.scatter(self.nodes[:,0],self.nodes[:,1],c=self.uCurrent[:,1])
                #plt.colorbar()
                #plt.xlabel("X")
                #plt.xlabel("Y")
                #plt.title("u current")
                #plt.show()


                self.uCurrent += unew

                #self.residual(iter)
                self.computeLoad(iter)
                #self.residual(iter)

                #extension1 = []
                #extension2 = [] 

                #for i in range(0,len(self.nodes)):
                #    if i in self.loadT or i in self.loadB :
                #        #extension.append(abs(iter*self.wCurrentExtension[i])-abs(self.uCurrent[i]))
                #        extension1.append(iter*abs(self.wCurrentExtension[i]))
                #        extension2.append(self.uCurrent[i])

                #print(np.linalg.norm(extension1),np.linalg.norm(extension2))
                #residual = np.linalg.norm(extension1) - np.linalg.norm(extension2) 
                #residual = np.linalg.norm(iter * self.wCurrent-extension)
                #residual = np.linalg.norm(extension)
                #residual = np.linalg.norm(self.b+self.f)
                residual = np.linalg.norm(self.uCurrent) - residual_old
                residual_old = np.linalg.norm(self.uCurrent)
                print("Iteration ",it," Residual: ",residual)
                it += 1

                #if it == 10 :
                #    break

            self.computeDamage()
            self.plot(iter)
            self.dump(iter)

    def plot(self,iter):
        step = iter + self.iter
        # Plot u_x
        plt.scatter(self.nodes[:,0]+self.uCurrent[:,0],self.nodes[:,1]+self.uCurrent[:,1],c=abs(self.uCurrent[:,0]))
        ax = plt.gca()
        ax.set_facecolor('#F0F8FF')
        v = np.linspace(min(abs(self.uCurrent[:,0])), max(abs(self.uCurrent[:,0])), 10, endpoint=True)
        clb = plt.colorbar(ticks=v)
        clb.set_label(r'Displacement $ u_x $',labelpad=5)
        plt.xlabel("Position $x$")
        plt.ylabel("Position $y$")
        plt.savefig("bond-based-2d-plate-u-x-"+str(self.h)+"-"+str(self.delta_factor)+"-"+str(step)+"-hard.pdf",bbox_inches='tight')
        plt.clf()
        # Plot u_y
        plt.scatter(self.nodes[:,0]+self.uCurrent[:,0],self.nodes[:,1]+self.uCurrent[:,1],c=self.uCurrent[:,1])
        ax = plt.gca()
        ax.set_facecolor('#F0F8FF')
        v = np.linspace(min(self.uCurrent[:,1]), max(self.uCurrent[:,1]), 10, endpoint=True)
        clb = plt.colorbar(ticks=v)
        clb.set_label(r'Displacement $ u_y $',labelpad=5)
        plt.xlabel("Position $x$")
        plt.ylabel("Position $y$")
        plt.savefig("bond-based-2d-plate-u-y-"+str(self.h)+"-"+str(self.delta_factor)+"-"+str(step)+"-hard.pdf",bbox_inches='tight')
        plt.clf()
        # Plot damage
        plt.scatter(self.nodes[:,0]+self.uCurrent[:,0],self.nodes[:,1]+self.uCurrent[:,1],c=self.d)
        ax = plt.gca()
        ax.set_facecolor('#F0F8FF')
        v = np.linspace(min(self.d), max(self.d), 10, endpoint=True)
        clb = plt.colorbar(ticks=v)
        clb.set_label(r'Damage',labelpad=5)
        plt.xlabel("Position $x$")
        plt.ylabel("Position $y$")
        plt.savefig("bond-based-2d-plate-d-"+str(self.h)+"-"+str(self.delta_factor)+"-"+str(step)+"-hard.pdf",bbox_inches='tight')
        plt.clf()

    def dump(self,iter):
        step = iter + self.iter
        filehandler = open("bond-based-2d-plate-"+str(self.h)+"-"+str(self.delta_factor)+"-"+str(step)+"-displacement-hard.npy", "wb")
        np.save(filehandler, self.uCurrent)
        filehandler = open("bond-based-2d-plate-"+str(self.h)+"-"+str(self.delta_factor)+"-"+str(step)+"-damage-hard.npy", "wb")
        np.save(filehandler,self.d)
        filehandler = open("bond-based-2d-plate-"+str(self.h)+"-"+str(self.delta_factor)+"-"+str(step)+"-f-hard.npy", "wb")
        np.save(filehandler,self.f)


if __name__=="__main__": 

    c = Compute(float(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
    c.solve(500,1e-5)
