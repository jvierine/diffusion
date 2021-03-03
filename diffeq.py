#!/usr/bin/env python3

import matplotlib

SMALL_SIZE = 14
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

import numpy as n
import matplotlib.pyplot as plt
import scipy.signal as s
import scipy.optimize as so
import scipy.io as sio

def diffusion_theory(u_m,                           # these are the measurements
                     t_meas,                        # measurement times
                     missing_idx=[],
                     k=1.0,                         # diffusion coefficient
                     t_model=n.linspace(0,5,num=100),   # time for model
                     sigma=0.01,                    # u_m measurement noise standard deviation
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
    n_meas=len(t_meas)
    n_model = len(t_model)
    if n_model > 2000:
        print("You are using a lot of model points. In order to be able to solve this problem efficiently, the number of model points should be <2000")
    
    A = n.zeros([n_meas + n_model + n_model-2,n_model*2])
        
    m = n.zeros(n_meas + n_model + n_model-2)
    dt = n.diff(t_model)[0]
    
    # diffusion equation
    # L is a very large number to ensure that the differential equation solution is nearly exact
    # this assentially means that these rows with L are equal to zero with a very very variance.
    # L >> 2*dt*1.0/n.min(sigma)
    L=2*dt*1e5/n.min(sigma)

    for i in range(n_model):
        m[i]=0.0

        # these two lines are k(u_a(t) - u_m(t))
        A[i,i]=k*L       # u_a(t[i])  
        A[i,i+n_model]=-k*L  # u_m(t[i])
        
        # this is the derivative -du_m(t)/d_t,
        # we make sure this derivative operator is numerically time-symmetric everywhere it can be, and asymmetric
        # only at the edges
        if i > 0 and i < (n_model-1):
            # symmetric derivative if not at edge
            # this cancels out
            #A[i,i+n_t]+=-L*0.5/dt       # -0.5 * (u_m(t)-u_m(t-dt))/dt           
            A[i,i+n_model-1]+=L*0.5/dt 
            A[i,i+n_model+1]+=-L*0.5/dt     # -0.5 * (u_m(t+dt)-u_m(t))/dt
            # this cancels out
            #A[i,i+n_t]+=L*0.5/dt
        elif i == n_model-1:
            # at edge, the derivative is not symmetric
            A[i,i+n_model]+=-L*1.0/dt                
            A[i,i+n_model-1]+=L*1.0/dt
        elif i == 0:
            # at edge, the derivative is not symmetric
            A[i,i+n_model]+=L*1.0/dt                
            A[i,i+n_model+1]+=-L*1.0/dt
                    
    # measurements u_m(t_1) ... u_m(t_N)
    # weight based on error standard deviation
    idx=n.arange(n_model,dtype=n.int)
    for i in range(n_meas):
        if i not in missing_idx:
            # linear interpolation between model points
            dist=n.abs(t_model - t_meas[i])
            w=(dist<dt)*(1-dist/dt)/sigma[i]
            A[i+n_model,idx+n_model] = w
            m[i+n_model]=u_m[i]/sigma[i]

    # smoothness regularization using tikhonov 2nd order difference
    for i in range(n_model-2):
        A[i+n_model+n_meas,i+0] =      smoothness/dt
        A[i+n_model+n_meas,i+1] = -2.0*smoothness/dt
        A[i+n_model+n_meas,i+2] =      smoothness/dt
        m[i+n_model+n_meas]=0.0
        
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

def unit_step_test(k=1.0,
                   missing_meas=False,
                   missing_t=[4,6],
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
                           t_meas=t,
                           missing_idx=missing_idx,
                           k=k,
                           t_model=t,
                           sigma=noise_std,
                           smoothness=1.0)


    xhat=n.linalg.lstsq(A,m_v)[0]

    u_a_estimate=xhat[0:n_t]
    u_m_estimate=xhat[n_t:(2*n_t)]

    Sigma_p=n.linalg.inv(n.dot(n.transpose(A),A))
    u_a_std=n.sqrt(n.diag(Sigma_p)[0:n_t])

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
    
def estimate_concentration(u_m,u_m_stdev,t_meas,k,n_model=400,smoothness=1e-5):
    n_meas = len(t_meas)

    # how many grid points do we have in the model
    t_model=n.linspace(n.min(t_meas),n.max(t_meas),num=n_model)
    
    A,m_v=diffusion_theory(u_m,k=k,t_meas=t_meas,t_model=t_model,sigma=u_m_stdev,smoothness=smoothness)
    
    # least squares solution
    xhat=n.linalg.lstsq(A,m_v)[0]    
    
    u_a_estimate=xhat[0:n_model]
    u_m_estimate=xhat[n_model:(2*n_model)]    
    
    # a posteriori error covariance
    Sigma_p=n.linalg.inv(n.dot(n.transpose(A),A))

    # standard deviation of estimated concentration u_a(t)
    std_p=n.sqrt(n.diag(Sigma_p))
    u_a_std=std_p[0:n_model]
    u_m_std=std_p[n_model:(2*n_model)]
    return(u_a_estimate, u_m_estimate, t_model, u_a_std, u_m_std)
    

def sensor_example():
    # read lab data
    d=sio.loadmat("time.mat")
    t=d["time"][0]
    u_m_meas=d["slowsens"][0]
    u_a_fast=d["fastsens"][0]

    # error standard deviation
    sigma=(0.001*n.abs(u_m_meas) + 0.1)*4.0

    k=(60.0*24.0)/30.0

    u_a_estimate, u_m_estimate, t_model, u_a_std, u_m_std= estimate_concentration(u_m_meas, sigma, t, k, n_model=400, smoothness=1e-5)
    
    plt.plot(t,u_m_meas,label="Slow sensor $u_m(t)$")
    plt.plot(t,u_a_fast,label="Fast sensor $\\hat{u}_a(t)$")
    plt.plot(t_model,u_a_estimate,label="Slow sensor $\\hat{u}_a(t)$")
    plt.plot(t_model,u_a_estimate+u_a_std*2,color="lightgreen")
    plt.plot(t_model,u_a_estimate-u_a_std*2,color="lightgreen")
    plt.ylabel("Concentration")
    plt.xlabel("Time (days)")    
    plt.legend()
    plt.show()

    
if __name__ == "__main__":

    sensor_example()
    unit_step_test(pfname="unit_step.png")    
    unit_step_test(missing_meas=True,pfname="unit_step_missing.png")    



        
        
    
