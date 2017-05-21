#Find the approximate interval of slice sampling
def _find_slice_interval(f, x,i,u, D, r, w=1):
    """Return approximated interval under f at height u.
    
    Parameters
    =========================================
    
    f: log_pdf of density to be sampled from
    x: a list of hyperparameters
    i: index of the hyperparameter to sample from
    u: uniform(0,1)*initial functional evaluation
    D: support of the distribution
    r,w: weights to narrow down the bracket at the beginning
    
    Returns
    =====================================================
    a,b: approximate interval of the bracket
    a_out,b_out: boundary of the bracket(two endpoints)
    """
    ups=np.zeros(len(x))
    ups[i]=(1-r)*w
    downs=np.zeros(len(x))
    downs[i]-r*w
    a=x+downs
    b=x+ups
   
    a_out = [a[i]]
    b_out = [b[i]]
    if a[i] < D[0]:
        a[i] = D[0]
        a_out[-1]= a[i]
    else:
        while f(a) > u:
            a[i] -= w
            a_out.append(a[i])
            if a[i] < D[0]:
                a[i] = D[0]
                a_out[-1] = a[i]
                break
    if b[i] > D[1]:
        b[i] = D[1]
        b_out[-1] = b[i]
    else:
        while f(b) > u:
            b[i] += w
            b_out.append(b[i])
            if b[i] > D[1]:
                b[i] = D[1]
                b_out[-1] = b[i]
                break
    return a, b, r, a_out, b_out
    
    
# Multivariate_slice_sample
def multivariate_slice_sample(logpdf_targets,hyps,D,num_samples=1,burn=30,lag=2):
    '''Perform multivariate slice sampling using a univariate slice sampling method
       iteratively updating each parameter in a manner like Gibbs sampling
       
       Parameters
       --------------------
       logpdf_targets: function      Target distribution
       hyps: list/array_like         a list of initial hyperparameters 
       D: list/array_like            A list of support for different hyperparameters
       num_samples: int              Number of samples
       burn : int, optional          Number of samples to discard before any are collected, default 1.
       lag : int, optional           Number of moves between successive samples, default 1.

       Returns
       -----------------
       A list of samples for all parameters
    '''
    #hyps: list of hyperparameters, must be initiated 
    n=0
    samples =[]
    while len(samples)<num_samples:
        hyps[n%len(hyps)]=slice_sample(hyps,logpdf_targets,n%len(hyps),D[n%len(hyps)])[0][n%len(hyps)]
        n+=1
        if n%len(hyps)==0 and burn <= (n-n%len(hyps))/len(hyps) and n%lag == 0:
            samples.append(hyps[:])
    return samples
    
#Vanilla Slice_sampling, adjusted for the multivariate sampling
def slice_sample(x_start, logpdf_target,i,D,num_samples=1,burn=1, lag=1,w=1.0,rng=None,):
    """Slice samples from the univariate disitrbution logpdf_target.
    Parameters
    ----------
    x_start : float
        Initial point.
    logpdf_target : function(x)
        Evaluates the log pdf of target distribution at x.
    D : tuple<float, float>
        Support of target distribution.
    num_samples : int, optional
        Number of samples to return, default 1.
    i: the index of the parameter to be sampled from
    rng : np.random.RandomState, optional
        Source of random bits.
    Returns
    -------
    samples : list
        `num_samples` length list.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    M = {'u':[], 'r':[], 'a_out':[], 'b_out':[], 'x_proposal':[], 'samples':[]}
    x = x_start
    num_iters = 0
    while len(M['samples']) < num_samples:
        num_iters += 1
        u = np.log(rng.rand()) + logpdf_target(x)
        r = rng.rand()
        a, b, r, a_out, b_out = _find_slice_interval(
            logpdf_target, x,i,u, D, r, w=w)
        #print(x)
        #print(a[0],b[0])
        x_proposal = []
        x_prime=x
        while True:
            newpara= rng.uniform(a[i], b[i])
            adds = np.zeros(len(x))
            adds[i]=newpara-x[i]
            x_prime=x+adds
            x_proposal.append(x)
            
            if logpdf_target(x_prime) > u:
                x = x_prime
                
                break
            else:
                if x_prime[i] > x[i]:
                    
                    b = x_prime
                else:
                    a = x_prime

        if burn <= num_iters and num_iters%lag == 0:
            M['u'].append(u)
            M['r'].append(r)
            M['a_out'].append(a_out)
            M['b_out'].append(b_out)
            M['x_proposal'].append(x_proposal)
            M['samples'].append(x)

    return M['samples']
