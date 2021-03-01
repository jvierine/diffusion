#!/usr/bin/env python3

import numpy as n
import matplotlib.pyplot as plt
import scipy.signal as s
import scipy.optimize as so


def diffusion_theory(u_m,                           # these are the measurements
                     missing_idx=[],
                     k=1.0,                         # diffusion coefficient
                     t=n.linspace(0,5,num=100),     # time  
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
    n_t = len(t)
    A = n.zeros([n_t*2 + n_t-2,n_t*2])
    m = n.zeros(n_t*2 + n_t-2)
    dt = n.diff(t)[0]
    
    # diffusion equation
    # L is a very large number to ensure that the differential equation solution is nearly exact
    # this assentially means that these rows with L are equal to zero with a very very variance. 
    L=1e6
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
        A[i+2*n_t,i+0] = smoothness/dt
        A[i+2*n_t,i+1] = -2.0*smoothness/dt
        A[i+2*n_t,i+2] = smoothness/dt
        m[i+2*n_t]=0.0
        
    # return theory matrix
    return(A,m)

def test_ua(t,t_on=1.0,u_a0=2.0,u_a1=2.0):
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
    noise_std = u_m*0.02 + 0.001
    m=u_m + noise_std*n.random.randn(len(u_m))
    return(m,noise_std)

def unit_step_test(k=0.5):
    t=n.linspace(0,10,num=100)
    u_a=test_ua(t,t_on=1.0,u_a0=2.0,u_a1=2.0)
    n_t=len(t)

    # simulate measurement affected by diffusion
    u_m=forward_model(t,u_a,k=k)
    m,noise_std=sim_meas(t,u_a,k=k)

    # create theory matrix
    missing_idx=n.arange(40,60,dtype=n.int)
    A,m_v=diffusion_theory(m,missing_idx=missing_idx,k=k,t=t,sigma=noise_std,smoothness=0.5)

#    xhat=n.linalg.lstsq(A,m_v)[0]
    xhat=so.nnls(A,m_v)[0]    

    u_a_estimate=xhat[0:n_t]
    u_m_estimate=xhat[n_t:(2*n_t)]
    
    # a posteriori error covariance
    Sigma_p=n.linalg.inv(n.dot(n.transpose(A),A))

    u_a_std=n.sqrt(n.diag(Sigma_p)[0:n_t])

    plt.plot(t,u_a,label="True $u_a(t)$",color="orange")
    plt.plot(t,u_a_estimate,color="blue",label="Estimate $\\hat{u}_a(t)$")
    plt.plot(t,u_a_estimate+2.0*u_a_std,color="lightblue",label="error bar")
    plt.plot(t,u_a_estimate-2.0*u_a_std,color="lightblue")
    plt.plot(t,u_m_estimate,label="Estimate $\\hat{u}_m(t)$",color="red")
    idx=n.arange(len(t),dtype=n.int)
    idx=n.setdiff1d(idx,missing_idx)
    plt.plot(t[idx],m[idx],".",label="Measurement $u_m(t)$",color="red")
    plt.plot(t,u_m,label="True $u_m(t)$",color="brown")    
    
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    plt.show()

def long_test():
    t,u_a=test_long_signal()
    n_t=len(t)

    u_m=forward_model(t,u_a)
    m,noise_std=sim_meas(t,u_a)

    # create theory matrix
    A,m_v=diffusion_theory(m,k=1.0,t=t,sigma=noise_std,smoothness=10.0)

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
    plt.show()

    
if __name__ == "__main__":
    unit_step_test()
#    long_test()


        
        
    
