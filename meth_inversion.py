#!/usr/bin/env python
#
# Demonstrate deconvolution of the diffusion equation using MCMC
#
import numpy as n
import matplotlib.pyplot as plt
import scipy.interpolate as sint
import scipy.optimize as so
import scam

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

def initial_guess(t,u_m,tau,N=5):
    """ Milosevich et.al., 2004 growth-law closed form solution """
    dt=n.diff(t)[0]
    X=n.exp(-dt/tau)
    u_a=n.zeros(len(t))
    for i in range(1,len(t)):
        u_a[i]= (u_m[i]-u_m[i-1]*X)/(1-X)
    u_a=n.convolve(n.repeat(1.0/N,N),u_a,mode="same")
    u_a[u_a<0]=0.0
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
    noise_std = u_m*0.003 + 0.0001
    m=u_m + noise_std*n.random.randn(len(u_m))
    return(m,noise_std)

def parameterized_model(t_nodes,u_an,t,u_m0=0.0,tau=1.0):
    # using a sparsely sampled u_a to simulate measurements
    # interpolate sparse u_a
    u_af = sint.interp1d(t_nodes,u_an)
    u_a=u_af(t)
    u_m=forward_model(t,u_a,tau=tau,u_m0=u_m0)
    return(u_a,u_m)

def run_mcmc(meas,t,t_nodes,u_a_nodes0,noise_std,tau=1.0, thin=100, n_samples=1000, smoothness=100.0):
    # t is measurement times
    # t_nodes is the points where we model the concentration
    # u_a_nodes is the initial guess for concentration at times t_nodes
    # meas is the measured concentration
    # smoothness is 2nd order difference regularization using an L1-metric. The larger the value, the more smooth we assume the solution to be 

    n_par=len(t_nodes)

    non_neg = True
    def ss(x):
        u_a_nodes=x
        u_a,u_m=parameterized_model(t_nodes,u_an=u_a_nodes,t=t,tau=tau)
        # negative log likelihood (variance weighted sum of squares)
        s=n.sum((1.0/(2.0*noise_std**2.0))*n.abs(u_m - meas)**2.0)
        # regularize for smoothness of solution (Total-Variation regularization for second order derivative)
        s+=n.sum(smoothness*n.abs(n.diff(n.diff(u_a_nodes))))

        # if one value is negative, the model is infinitely unlikely
        if non_neg and n.sum(u_a_nodes < 0.0) > 0:
            # non-negative prior
            return(1e99)
        print("sum of squares %1.2f"%(s))
        return(s)

    # fmin search first a few times (non_negativity doesn't work well with nelder-mead)
    xhat=so.fmin(ss,u_a_nodes0)

    # use MCMC to sample parameters from the likelihood distribution
    chain=scam.scam(ss,x0=xhat,n_par=n_par,step=n.repeat(0.02,n_par),n_iter=n_samples,thin=thin)

    # only use second half of the chain. the first part is thrown away, as the chain may have not converged yet
    chain=chain[int(chain.shape[0]/2):(chain.shape[0]),:]
    
    return(chain)

if __name__ == "__main__":
    # simulate measurements
    t=n.linspace(0,5,num=200)
    u_a=test_ua(t)
    u_m=forward_model(t,u_a)
    m,noise_std=sim_meas(t,u_a)

    # Use the Milosevich et.al., 2004 filtered closed form solution as initial guess
    u_a_guess=initial_guess(t,m,tau=1.0)
    plt.title("Initial guess")
    plt.plot(t,u_a_guess)
    plt.plot(t,u_a)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.show()

    # model u_a at these time points
    # don't use same grid for model as we used for simulating measurements
    n_nodes=180
    t_nodes=n.linspace(0,5,num=n_nodes)
    # setup initial guess
    u_a_guess_f=sint.interp1d(t,u_a_guess)
    u_a_nodes=u_a_guess_f(t_nodes)

    # Sample values of u_a using MCMC
    chain=run_mcmc(m,t,t_nodes,u_a_nodes,noise_std)
    
    # maximum a posteriori estimate
    max_apost_par=n.mean(chain,axis=0)
    # 2*standard deviation error bars
    max_apost_std=2.0*n.std(chain,axis=0)    
    u_a_ml,u_m_ml=parameterized_model(t_nodes,u_an=max_apost_par,t=t)

    plt.title("Diffusion inversion")
    plt.plot(t,u_a,label="True $u_a(t)$")
    plt.plot(t,u_a_ml,label="MAP Estimate $\\hat{u}_a(t)$")
    plt.errorbar(t_nodes,max_apost_par,yerr=max_apost_std)        
    plt.plot(t,u_m_ml,label="MAP Model $\\hat{u}_m(t)$")        
    plt.plot(t,m,".",label="Measurement+noise $u_m(t)+\\xi(t)$")
    plt.plot(t,u_m,label="Measurement $u_m(t)$")    
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    plt.show()




    
