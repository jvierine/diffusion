#!/usr/bin/env python3

import numpy as n
import matplotlib.pyplot as plt


def diffusion_theory(u_m,                           # these are the measurements
                     k=1.0,                         # diffusion coefficient
                     t=n.linspace(0,5,num=100),     # time  
                     sigma=0.01,                    # u_m measurement noise standard deviation      
                     smoothness=1.0):                   
    #
    # 0 = k (u_a(t) - u_m(t)) - d u_m / dt |_t
    # [0;   
    #  u_m ] = A x 
    #
    # x = [ u_a(t_1),
    #       u_a(t_2),
    #       ...
    #       u_a(t_N),
    #       u_m(t_1),
    #       u_m(t_2),
    #       ...
    #       u_m(t_N) ]
    n_t = len(t)
    A = n.zeros([n_t*2 + n_t-2,n_t*2])
    m = n.zeros(n_t*2 + n_t-2)
    dt = n.diff(t)[0]
    
    # diffusion equation
    # L is a very large number to ensure that the differential equation solution is nearly exact
    L=1e6
    for i in range(n_t):
        m[i]=0.0
        # boundary condition u_a(t_1) - u_m(t_1) = 0
        if i == 0:
            A[i,i]=1.0*L
            A[i,i+n_t]=-1.0*L
        else:
            if i < n_t:
                A[i,i]=k*L       # u_a(t[i])
                A[i,i+n_t]=-k*L  # u_m(t[i])
                # symmetric derivative if not at edge
                if i < (n_t-1):                
                    A[i,i+n_t]+=-L*0.5/dt                
                    A[i,i+n_t-1]+=L*0.5/dt
                    A[i,i+n_t+1]+=-L*0.5/dt                                
                    A[i,i+n_t]+=L*0.5/dt
                else:
                # at edge, the derivative is symmetric
                    A[i,i+n_t]+=-L*1.0/dt                
                    A[i,i+n_t-1]+=L*1.0/dt
                    
    # measurements u_m(t_1) ... u_m(t_N)
    # weight based on error standard deviation
    for i in range(n_t):
        A[i+n_t,i+n_t] = 1.0/sigma[i]
        m[i+n_t]=u_m[i]/sigma[i]

    # smoothness regularization using tikhonov 2nd order difference
    for i in range(n_t-2):
        A[i+2*n_t,i+0] = smoothness
        A[i+2*n_t,i+1] = -2.0*smoothness
        A[i+2*n_t,i+2] = smoothness
        m[i+2*n_t]=0.0
        
    # return theory matrix
    return(A,m)

def test_ua(t,t_on=1.0,u_a0=2.0,u_a1=0.0):
    """ simulated measurement """
    # this is a simulated true instantaneous concentration
    # simple "on" at t_on model and gradual decay
    u_a=n.zeros(len(t))
    u_a=n.linspace(u_a0,u_a1,num=len(t))
    # turn "on"
    u_a[t<t_on]=0.0
    # smooth the step a little bit
    u_a=n.real(n.fft.ifft(n.fft.fft(n.repeat(1.0/5,5),len(u_a))*n.fft.fft(u_a)))
    u_a[t<t_on]=0.0    
    return(u_a)

def forward_model(t,u_a,tau=1.0,u_m0=0.0):
    """ forward model """
    # evaluate the forward model, which includes slow diffusion
    # t is time
    # u_a is the concentration
    # tau is the diffusion time constant
    # u_m0 is the initial boundary condition for the diffused quantity
    u_m = n.zeros(len(t))
    dt = n.diff(t)[0]
    for i in range(1,len(t)):
        u_m[i]=u_a[i] - (u_a[i]-u_m[i-1])*n.exp(-dt/tau)
    return(u_m)

def sim_meas(t,u_a,tau=1.0,u_m0=0.0):
    # simulate measurements, including noise
    u_m=forward_model(t,u_a,tau=tau,u_m0=u_m0)
    # a simple model for measurement noise, which includes
    # noise that is always there, and noise that depends on the quantity
    noise_std = u_m*0.003 + 0.001
    m=u_m + noise_std*n.random.randn(len(u_m))
    return(m,noise_std)

if __name__ == "__main__":
    n_t=200
    t=n.linspace(0,5,num=n_t)

    u_a=test_ua(t)
    u_m=forward_model(t,u_a)
    m,noise_std=sim_meas(t,u_a)

    # create theory matrix
    A,m_v=diffusion_theory(m,k=1.0,t=t,sigma=noise_std,smoothness=10.0)

    xhat=n.linalg.lstsq(A,m_v)[0]

    u_a_estimate=xhat[0:n_t]
    
    # a posteriori error covariance
    Sigma_p=n.linalg.inv(n.dot(n.transpose(A),A))

    u_a_std=n.sqrt(n.diag(Sigma_p)[0:n_t])
          
    plt.plot(t,m,".",label="Measurement $u_m(t)$",color="red")
    plt.plot(t,u_a,label="True $u_a(t)$",color="orange")
    plt.plot(t,u_a_estimate,color="blue",label="Estimate $\\hat{u}_a(t)$")
    plt.plot(t,u_a_estimate+2.0*u_a_std,color="lightblue",label="error bar")
    plt.plot(t,u_a_estimate-2.0*u_a_std,color="lightblue")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    plt.show()

        
        
    
