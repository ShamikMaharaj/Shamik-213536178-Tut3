#Shamik Maharaj
#213536178
#Tutorial3

#Q1================================================================

import numpy
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as ani


class NBody:
      def __init__(self,x=0,y=0,m=1):
          self.m=m
          self.x=x
          self.y=y
          self.conf={"NoParticles":3,"G":1,"e":0.03}

      def PE(self):
          potential=0
          for n in range(self.conf["NoParticles"]):
              x=self.x[n]-self.x
              y=self.y[n]-self.y
              r_2=x**2+y**2
              soft=self.conf["e"]**2
              r_2[r_2<soft]=soft
              r_2=r_2+self.conf["e"]**2
              r=numpy.sqrt(r_2)
              potential=potential+self.conf["G"]*numpy.sum(self.m/r)*self.m[n]
          return -0.5*potential

#Q2==========================================================
      def init(self):
          self.x=numpy.random.randn(self.conf["NoParticles"])
          self.y=numpy.random.randn(self.conf["NoParticles"])
          self.m=numpy.ones(self.conf["NoParticles"])*self.m/self.conf["NoParticles"]
          self.vx=numpy.zeros(self.conf["NoParticles"])
          self.vy=numpy.zeros(self.conf["NoParticles"])
          self.fx=numpy.zeros(self.conf["NoParticles"])
          self.fy=numpy.zeros(self.conf["NoParticles"])

      def force(self):
          for n in range(self.conf["NoParticles"]):
              x=self.x[n]-self.x
              y=self.y[n]-self.y
              r_2=x**2+y**2
              soft=self.conf["e"]**2
              r_2[r_2<soft]=soft
              r_2=r_2+self.conf["e"]**2
              r=numpy.sqrt(r_2)
              r_3=(r*r_2)
              self.fx[n]=-self.conf["G"]*numpy.sum((self.m*x)/r_3)*self.m[n]
              self.fy[n]=-self.conf["G"]*numpy.sum((self.m*y)/r_3)*self.m[n]

      def updateParticles(self,timestep):
          self.x=self.x+self.vx*timestep
          self.y=self.y+self.vy*timestep
          potential=self.PE()
          self.force()
          self.vx=self.vx+self.fx*timestep
          self.vy=self.vy+self.fy*timestep
          kin_energy=0.5*numpy.sum(self.m*(self.vx**2+self.vy**2))
          return potential+kin_energy

system=NBody()
system.init()
plt.plot(system.x,system.y,'.')
plt.show()
timestep=0.01
oversamp=5
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-8, 8), ylim=(-8, 8))
line, = ax.plot([], [], '*', lw=2)

energy=numpy.array([])
print(type(energy))
def Animate(crud):
    global system,line
    for ii in range(oversamp):
        value=numpy.array(system.updateParticles(timestep))
    numpy.append(energy,value)
    print(value)
    line.set_data(system.x,system.y)

ani = ani.FuncAnimation(fig, Animate, numpy.arange(30),interval=25, blit=False)
plt.show()

time=numpy.range(timestep*energy.shape[0],timestep)
plt.plot(time,energy)
plt.set_xlabel("Time(s)")
plt.set_ylabel("Energy")
plt.show()

#Q3===========================================================


def fit(x,n,n):
    a=numpy.zeros([x.shape[0],n])
    a[:,0]=1
    for n in range(1,n):
        a[:,n]=a[:,n-1]*x
    a=numpy.matrix(a)
    d=numpy.matrix(n).transpose()
    lhs=a.transpose()*a
    rhs=a.transpose()*d
    fitp=numpy.linalg.inv(lhs)*rhs
    p=a*fitp
    return p

n=100
x=numpy.linspace(0,2*numpy.pi,n)
y=numpy.sin(x)
z=numpy.cos(x)
ny=y+numpy.random.randn(n)
nz=z+numpy.random.randn(n)
n=5

py=fit(x,n,ny)
p_z=fit(x,n,nz)

plt.plot(x,points_y,"X",label="sine")
plt.plot(x,py,"o",label="Fit")
plt.legend()
plt.show()

plt.plot(x,points_z,"X",label="cosine")
plt.plot(x,pz,"o",label="Fit")
plt.legend()
plt.show()
 
#Q4=======================================================

def gaussian(t,s=0.5,a=1,c=0):
    dat=numpy.exp(-0.5*(t-c)**2/s**2)*a
    dat+=numpy.random.randn(t.size)
    return dat

def lorentzian(t,a=1,b=1,c=0):
    dat=a/(b+(t-c)**2)
    dat+=numpy.random.randn(t.size)
    return dat


def dummyOffset(s):
    return s*numpy.random.randn(s.size)

class Gaussian:
    def init(self,t,s=0.5,a=1.0,c=0,offset=0):
        self.t=t
        self.y=gaussian(t,s,a,c)+offset
        self.err=numpy.ones(t.size)
        self.s=s
        self.a=a
        self.c=c
        self.offset=offset

    def chisq(self,vec):
        s=vec[0]
        a=vec[1]
        c=vec[2]
        off=vec[3]

        pred=off+a*numpy.exp(-0.5*(self.t-c)**2/s**2)
        chisq=numpy.sum((self.y-pred)**2/self.err**2)
        return chisq

class Lorentzian:
      def init(self,t,a=1,b=1,c=0,offset=0):
          self.t=t
          self.y=lorentzian(t,a,b,c)+offset
          self.err=numpy.ones(t.size)
          self.a=a
          self.b=b
          self.c=c
          self.offset=offset

      def chisq(self,vec):
          a=vec[0]
          b=vec[1]
          c=vec[2]
          off=vec[3]
  
          pred=off+a/(b+(self.t-c)**2)
          chisq=numpy.sum((self.y-pred)**2/self.err**2)
          return chisq

def mcmc(data,start_pos,nstep,scale=None):
    nparam=start_pos.size
    params=numpy.zeros([nstep,nparam+1])
    params[0,0:-1]=start_pos
    cur_chisq=data.chisq(start_pos)
    cur_pos=start_pos.copy()
    if scale==None:
        scale=numpy.ones(nparam)
    for i in range(1,nstep):
        new_pos=cur_pos+dummyOffset(scale)
        new_chisq=data.chisq(new_pos)
        if new_chisq<cur_chisq:
            accept=True
        else:
            delt=new_chisq-cur_chisq
            prob=numpy.exp(-0.5*delt)
            if numpy.random.rand()<prob:
                accept=True
            else:
                accept=False
        if accept: 
            cur_pos=new_pos
            cur_chisq=new_chisq
        params[i,0:-1]=cur_pos
        params[i,-1]=cur_chisq
    return params


if __name__=='__main__':
    
    
    t=numpy.arange(-5,5,0.01)
    dat=Gaussian(t,a=2.5)

    guess=numpy.array([0.3,1.2,0.3,-0.2])
    scale=numpy.array([0.1,0.1,0.1,0.1])
    nstep=100000
    chain=mcmc(dat,guess,nstep,scale)
    nn=numpy.round(0.2*nstep)
    chain=chain[int(nn):,:]
    
    param_true=numpy.array([dat.s,dat.a,dat.c,dat.offset])
    print("For the gaussian:  ")
    for i in range(0,param_true.size):
        val=numpy.mean(chain[:,i])
        scat=numpy.std(chain[:,i])
        print [param_true[i],val,scat]

    print("For the lorentzian")
    dat_lor=Lorentzian(t)
    chain_lor=mcmc(dat_lor,guess,nstep,scale)
    chain_lor=chain_lor[int(nn):,:]

    param_true=numpy.array([dat_lor.a,dat_lor.b,dat_lor.c,dat_lor.offset])
    for i in range(0,param_true.size):
        val=numpy.mean(chain[:,i])
        scat=numpy.std(chain[:,i])
        print [param_true[i],val,scat]
 
