import numpy as n

def scam(ss,x0,n_par=2,step=[0.01,0.01],n_iter=1000,thin=10,debug=False,update_period=1000):
    # single component adaptive metropolis-hastings (SCAM)

    accepted=n.zeros(n_par)
    all_try=n.zeros(n_par)    
    chain=n.zeros([n_iter,n_par],dtype=n.float32)
    for i in range(n_iter*thin):
        lp = ss(x0)
        xtry=n.copy(x0)
        pi = int(n.random.rand(1)*n_par)
        xtry[pi]+=n.random.randn(1)*step[pi]
        lp_try = ss(xtry)
        all_try[pi]+=1
        if lp_try <= lp:
            x0=xtry
            accepted[pi]+=1
        else:
            alpha=n.log(n.random.rand(1))
            if alpha < (lp - lp_try):
                x0=xtry
                accepted[pi]+=1

        # adapt proposal based on acceptance rate
        if i%update_period == 0 and i > 0:
            ratio=accepted/(all_try+1)
            low_ratio=n.where(ratio < 0.2)[0]
            high_ratio=n.where(ratio > 0.8)[0]
            step[low_ratio]=step[low_ratio]/2.0
            step[high_ratio]=step[high_ratio]*2.0
            print(step)
            accepted[:]=0.0
            all_try[:]=0.0
        chain[int(i/thin),:]=x0
        if debug:
            print(lp)
    return(chain)
