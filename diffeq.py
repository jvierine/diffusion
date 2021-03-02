#!/usr/bin/env python3

import matplotlib

SMALL_SIZE = 14
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

import numpy as n
import matplotlib.pyplot as plt
import scipy.signal as s
import scipy.optimize as so
import scipy.sparse as ss
import scipy.io as sio

def diffusion_theory(u_m,                           # these are the measurements
                     missing_idx=[],
                     k=1.0,                         # diffusion coefficient
                     t=n.linspace(0,5,num=100),     # time  
                     sigma=0.01,                    # u_m measurement noise standard deviation
                     sparse=False,
                     smoothness=1.0):                   
    #
    # Let's solve this equation:
    # 0 = k (u_a(t) - u_m(t)) - d u_m / dt
    #
    # Here u_a(t) is concentration, and u_m(t) is constration affected by diffusion
    #
    # We'll express this equation using a discretized linear equation:
    # m = A x
    #
    # m = [ 0 ,  # these first zeroes are to satisfy the growth law equation
    #       0 ,
    #       ...,
    #       0,    # n_t
    #       u_m(t_1)+noise_1, # these are measurements of u_m(t), including noise
    #       u_m(t_2)+noise_2,
    #       ...,
    #       u_m(t_N)+noise_N ] 
    #
    # The unknown parameters are:   
    # 
    # x = [ u_a(t_1),   # concentration u_a(t)
    #       u_a(t_2),
    #       ...
    #       u_a(t_N),
    #       u_m(t_1),   # concentration u_m(t)
    #       u_m(t_2),
    #       ...
    #       u_m(t_N) ]
    #
    # We'll also include regularization to the assumes d^2 u_n(t) / dt^2 is a small normal random variable with variance
    # corresponding to 1.0/smoothness^2.
    #
    #
    n_t = len(t)
    if sparse:
        A = ss.csc_matrix((n_t*2 + n_t-2,n_t*2), dtype=n.float64)
    else:
        A = n.zeros([n_t*2 + n_t-2,n_t*2])
        
    m = n.zeros(n_t*2 + n_t-2) 
    dt = n.diff(t)[0]
    
    # diffusion equation
    # L is a very large number to ensure that the differential equation solution is nearly exact
    # this assentially means that these rows with L are equal to zero with a very very variance.
    #    if sparse:
    #       # the iterative sparse solver can't cope with very large dynamic range
    #      L=10.0
    # else:
    # condition to ensure that the diffusion part is
    # weighted significantly more than the measurements
    # L >> 2*dt*1.0/n.min(sigma)
    if sparse:
        L=2*dt*1e2/n.min(sigma)
    else:
        L=2*dt*1e5/n.min(sigma)

    for i in range(n_t):
        m[i]=0.0

        # these two lines are k(u_a(t) - u_m(t))
        A[i,i]=k*L       # u_a(t[i])  
        A[i,i+n_t]=-k*L  # u_m(t[i])
        
        # this is the derivative -du_m(t)/d_t,
        # we make sure this derivative operator is numerically time-symmetric everywhere it can be, and asymmetric
        # only at the edges
        if i > 0 and i < (n_t-1):
            # symmetric derivative if not at edge
            # this cancels out
            #A[i,i+n_t]+=-L*0.5/dt       # -0.5 * (u_m(t)-u_m(t-dt))/dt           
            A[i,i+n_t-1]+=L*0.5/dt 
            A[i,i+n_t+1]+=-L*0.5/dt     # -0.5 * (u_m(t+dt)-u_m(t))/dt
            # this cancels out
            #A[i,i+n_t]+=L*0.5/dt
        elif i == n_t-1:
            # at edge, the derivative is not symmetric
            A[i,i+n_t]+=-L*1.0/dt                
            A[i,i+n_t-1]+=L*1.0/dt
        elif i == 0:
            # at edge, the derivative is not symmetric
            A[i,i+n_t]+=L*1.0/dt                
            A[i,i+n_t+1]+=-L*1.0/dt
                    
    # measurements u_m(t_1) ... u_m(t_N)
    # weight based on error standard deviation
    for i in range(n_t):
        if i not in missing_idx:
            A[i+n_t,i+n_t] = 1.0/sigma[i]
            m[i+n_t]=u_m[i]/sigma[i]

    # smoothness regularization using tikhonov 2nd order difference
    for i in range(n_t-2):
        A[i+2*n_t,i+0] =      smoothness/dt
        A[i+2*n_t,i+1] = -2.0*smoothness/dt
        A[i+2*n_t,i+2] =      smoothness/dt
        m[i+2*n_t]=0.0
        
    # return theory matrix
    return(A,m)

def test_ua(t,t_on=1.0,u_a0=1.0,u_a1=1.0):
    """ simulated measurement """
    # this is a simulated true instantaneous concentration
    # simple "on" at t_on model and gradual decay
    u_a=n.zeros(len(t))
    u_a=n.linspace(u_a0,u_a1,num=len(t))
    # turn "on"
#    u_a[t<t_on]=0.0
    # smooth the step a little bit
    u_a=n.real(n.fft.ifft(n.fft.fft(n.repeat(1.0/5,5),len(u_a))*n.fft.fft(u_a)))
    u_a[t<t_on]=0.0    
    return(u_a)

def test_long_signal(T_max=10,n_t=1000,n_spikes=30,spike_len=20,spike_amp=1.0):
    """ simulate a longer measurement """
    t=n.linspace(0,T_max,num=n_t)
    u_a=n.zeros(len(t))

    for i in range(n_spikes):
        t_spike = int(n_t*n.random.rand(1))
        u_a[t_spike]=n.abs(n.random.randn(1)*spike_amp)

    u_a=n.real(n.fft.ifft(n.fft.fft(n.repeat(1.0/float(spike_len),spike_len),len(u_a))*n.fft.fft(u_a)))
    u_a=n.real(n.fft.ifft(n.fft.fft(n.repeat(1.0/float(spike_len),spike_len),len(u_a))*n.fft.fft(u_a)))
    u_a=n.real(n.fft.ifft(n.fft.fft(n.repeat(1.0/float(spike_len),spike_len),len(u_a))*n.fft.fft(u_a)))    
    
    return(t,u_a)


def forward_model(t,u_a,k=1.0,u_m0=0.0):
    """ forward model """
    # evaluate the forward model, which includes slow diffusion
    # t is time
    # u_a is the concentration
    # k is the diffusion time constant
    # u_m0 is the initial boundary condition for the diffused quantity
    u_m = n.zeros(len(t))
    dt = n.diff(t)[0]
    for i in range(1,len(t)):
        u_m[i]=u_a[i] - (u_a[i]-u_m[i-1])*n.exp(-k*dt)
    return(u_m)

def sim_meas(t,u_a,k=1.0,u_m0=0.0):
    # simulate measurements, including noise
    u_m=forward_model(t,u_a,k=k,u_m0=u_m0)
    # a simple model for measurement noise, which includes
    # noise that is always there, and noise that depends on the quantity
    noise_std = u_m*0.03 + 0.001
    m=u_m + noise_std*n.random.randn(len(u_m))
    return(m,noise_std)

def calc_sparse_covariance(A,frac=0.99):
    # only use (1-frac) of the non-zero elements of F=n.transpose(A)*A
    # for inverting 
    n_t=int(A.shape[1]/2)
    # a posteriori error covariance
    F=A.transpose().dot(A).toarray()
    thresh=n.sort(F.flatten())
    thresh=thresh[int(len(thresh)*frac)]
    
    nonzero_mask = n.array(n.abs(F[F.nonzero()]) > thresh)[0]
    rows = F.nonzero()[0][nonzero_mask]
    cols = F.nonzero()[1][nonzero_mask]
#    print(len(rows))
    F2=ss.csr_matrix(F.shape, dtype=n.float)
    F2[rows, cols] = F[rows,cols]
    
    # ensure diagonal
    for i in range(F.shape[0]):
        F2[i,i]=F[i,i]
        if i>0:
            F2[i-1,i-1]=F[i-1,i-1]
        if i<(F.shape[0]-1):
            F2[i+1,i+1]=F[i+1,i+1]
        if i>1:
            F2[i-2,i-2]=F[i-2,i-2]
        if i<(F.shape[0]-2):
            F2[i+2,i+2]=F[i+2,i+2]            
                
    Sigma_p=ss.linalg.inv(F2).toarray()
    u_a_std=n.sqrt(n.diag(Sigma_p)[0:n_t])
    return(u_a_std)

def unit_step_test(k=1.0,
                   missing_meas=False,
                   missing_t=[4,6],
                   sparse=False,
                   pfname="unit_step.png"):
    
    t=n.linspace(0,10,num=500)
    
    if missing_meas:
        idx=n.arange(len(t))
        missing_idx=n.where( (t[idx]>missing_t[0]) & (t[idx]<missing_t[1]))[0]
    else:
        missing_idx=[]
    u_a=test_ua(t,t_on=2.0,u_a0=1.0,u_a1=1.0)
    n_t=len(t)

    # simulate measurement affected by diffusion
    u_m=forward_model(t,u_a,k=k)
    m,noise_std=sim_meas(t,u_a,k=k)

    # create theory matrix
    A,m_v=diffusion_theory(m,
                           missing_idx=missing_idx,
                           k=k,
                           t=t,
                           sigma=noise_std,
                           smoothness=1.0,
                           sparse=sparse)

    if sparse:
        xhat,p2,p3,p4,p5,p6,p7,p8,p9,x_var=ss.linalg.lsqr(A,m_v,atol=1e-13,btol=1e-13,calc_var=True)

    else:
        xhat=n.linalg.lstsq(A,m_v)[0]
        
    print("done lsqr")

    u_a_estimate=xhat[0:n_t]
    u_m_estimate=xhat[n_t:(2*n_t)]

    if sparse:
        u_a_std=calc_sparse_covariance(A,frac=0.95)
    else:
        Sigma_p=n.linalg.inv(n.dot(n.transpose(A),A))
        u_a_std=n.sqrt(n.diag(Sigma_p)[0:n_t])
        
    print("done sigma_p")    
    # sparsify
    #    vals=n.sort(n.abs(F.toarray().flatten()))
    #   thresh=vals[int(0.9*len(vals))]
    


    #    Sigma_p=n.linalg.inv(n.dot(n.transpose(A),A))
    #    u_a_std = n.sqrt(1.0/F.diagonal())[0:n_t]


    plt.plot(t,u_a,label="True $u_a(t)$",color="orange")
    plt.plot(t,u_m,label="True $u_m(t)$",color="brown")    
    plt.plot(t,u_a_estimate,color="blue",label="Estimate $\\hat{u}_a(t)$")
    plt.plot(t,u_a_estimate+2.0*u_a_std,color="lightblue",label="2-$\\sigma$ uncertainty")
    lower_bound=u_a_estimate-2.0*u_a_std
    lower_bound[lower_bound<0]=0.0
    plt.plot(t,lower_bound,color="lightblue")
    plt.plot(t,u_m_estimate,label="Estimate $\\hat{u}_m(t)$",color="red")
    idx=n.arange(len(t),dtype=n.int)
    idx=n.setdiff1d(idx,missing_idx)
    plt.plot(t[idx],m[idx],".",label="Measurement",color="red")

    
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend(ncol=2)
    plt.ylim([-0.2,3.0])
    plt.tight_layout()
    plt.savefig(pfname)
    
    plt.show()

def long_test():
    t,u_a=test_long_signal()
    n_t=len(t)

    u_m=forward_model(t,u_a)
    m,noise_std=sim_meas(t,u_a)

    # create theory matrix
    A,m_v=diffusion_theory(m,k=1.0,t=t,sigma=noise_std,smoothness=20.0)

#    xhat=n.linalg.lstsq(A,m_v)[0]
    # non-negative least-squares
    xhat=so.nnls(A,m_v)[0]
    
    u_a_estimate=xhat[0:n_t]
    u_m_estimate=xhat[n_t:(2*n_t)]    
    
    # a posteriori error covariance
    Sigma_p=n.linalg.inv(n.dot(n.transpose(A),A))

    u_a_std=n.sqrt(n.diag(Sigma_p)[0:n_t])
          

    plt.plot(t,u_m_estimate,label="Estimate $\\hat{u}_m(t)$",color="red")
    plt.plot(t,u_m,label="True measurement $u_m(t)$",color="brown")        
    plt.plot(t,u_a,label="True $u_a(t)$",color="orange")
    plt.plot(t,u_a_estimate,color="blue",label="Estimate $\\hat{u}_a(t)$")
    plt.plot(t,u_a_estimate+2.0*u_a_std,color="lightblue",label="error bar")
    plt.plot(t,u_a_estimate-2.0*u_a_std,color="lightblue")
    

    plt.plot(t,m,".",label="Measurement+noise $u_m(t)+\\xi(t)$",color="red",alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pfname)
    plt.show()

def sensor_data():
    # read lab data
    d=sio.loadmat("time.mat")
    t=d["time"][0]
    u_m_meas=d["slowsens"][0]
    u_a_fast=d["fastsens"][0]

    # error standard deviation
    sigma=(0.001*n.abs(u_m_meas) + 0.1)*2.0

    # figure out error standard deviation
    #plt.plot(n.diff(u_m_meas)/sigma[0:(len(u_m_meas)-1)])
    #plt.show()

    k=(60.0*24.0)/30.0
    n_meas = len(t)

    # undersample dataset
    decimation=10.0
    idx=n.array(n.floor(n.arange(n_meas/decimation)*decimation),dtype=n.int)
    t=t[idx]
    u_m_meas=u_m_meas[idx]
    u_a_fast=u_a_fast[idx]
    sigma=sigma[idx]
    
    n_t=len(idx)
    
    A,m_v=diffusion_theory(u_m_meas,k=k,t=t,sigma=sigma,smoothness=1e-4,sparse=False)
    # least squares solution
    xhat=n.linalg.lstsq(A,m_v)[0]    
    
    u_a_estimate=xhat[0:n_t]
    u_m_estimate=xhat[n_t:(2*n_t)]    
    
    # a posteriori error covariance
    Sigma_p=n.linalg.inv(n.dot(n.transpose(A),A))

    # standard deviation of estimated concentration u_a(t)
    u_a_std=n.sqrt(n.diag(Sigma_p)[0:n_t])
    
    plt.plot(t,u_m_meas,label="Slow sensor $u_m(t)$")
    plt.plot(t,u_a_fast,label="Fast sensor $\\hat{u}_a(t)$")
    plt.plot(t,u_a_estimate,label="Slow sensor $\\hat{u}_a(t)$")
    plt.plot(t,u_a_estimate+u_a_std*2,color="lightgreen")
    plt.plot(t,u_a_estimate-u_a_std*2,color="lightgreen")
    plt.ylabel("Concentration")
    plt.xlabel("Time (days)")    
    plt.legend()
    plt.show()


def sensor_data_sparse():
    # read lab data
    d=sio.loadmat("time.mat")
    t=d["time"][0]
    u_m_meas=d["slowsens"][0]
    u_a_fast=d["fastsens"][0]

    # error standard deviation
    sigma=(0.001*n.abs(u_m_meas) + 0.1)*2.0

    # figure out error standard deviation
    #plt.plot(n.diff(u_m_meas)/sigma[0:(len(u_m_meas)-1)])
    #plt.show()

    k=(60.0*24.0)/30.0

    # undersample dataset
    decimation=10.0
    n_meas = len(t)    
    idx=n.array(n.floor(n.arange(n_meas/decimation)*decimation),dtype=n.int)
    t=t[idx]
    u_m_meas=u_m_meas[idx]
    u_a_fast=u_a_fast[idx]
    sigma=sigma[idx]
    
    n_t=len(idx)

    n_t=len(t)

    # with a sparse matrix theory, we can use all measurements
    A,m_v=diffusion_theory(u_m_meas,k=k,t=t,sigma=sigma,smoothness=1e-4,sparse=True)
    
    # least squares solution
    # we need to ensure good convergence, so manually increase iter_lim
    #    xhat,p2,p3,p4,p5,p6,p7,p8,p9,x_var=ss.linalg.lsqr(A,m_v,atol=1e-30,btol=1e-30,show=True,iter_lim=100000)
    xhat=ss.linalg.lsmr(A,m_v,atol=1e-30,btol=1e-30,show=True,maxiter=100000)[0]
    
    u_a_estimate=xhat[0:n_t]
    u_m_estimate=xhat[n_t:(2*n_t)]

    plt.plot(t,u_m_meas,label="Slow sensor $u_m(t)$")
    plt.plot(t,u_a_fast,label="Fast sensor $\\hat{u}_a(t)$")
    plt.plot(t,u_a_estimate,label="Slow sensor $\\hat{u}_a(t)$")

    plt.ylabel("Concentration")
    plt.xlabel("Time (days)")    
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    sensor_data()
    # this will take a bit longer
    sensor_data_sparse()    
    unit_step_test(pfname="unit_step.png",sparse=False)
    unit_step_test(missing_meas=True,pfname="unit_step_missing.png")    
#    long_test()


        
        
    
