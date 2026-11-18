import numpy as np, pennylane as qml
def fmap(x, L, kind='zz'):
    n=len(x)
    for _ in range(L):
        for i in range(n):
            qml.Hadamard(wires=i); qml.RZ(2*x[i],wires=i)
        if kind=='zonly': continue
        pairs=[(i,(i+1)%n) for i in range(n)] if kind=='zz' else [(i,j) for i in range(n) for j in range(i+1,n)]
        if n==2 and kind=='zz': pairs=[(0,1)]
        for i,j in pairs:
            qml.CNOT(wires=[i,j]); qml.RZ(2*(np.pi-x[i])*(np.pi-x[j]),wires=j); qml.CNOT(wires=[i,j])
_cache={}
def _state_fn(n,L,kind):
    k=(n,L,kind)
    if k not in _cache:
        dev=qml.device('default.qubit',wires=n)
        @qml.qnode(dev)
        def f(x): fmap(x,L,kind); return qml.state()
        _cache[k]=f
    return _cache[k]
def states(X,L,kind='zz'):
    f=_state_fn(X.shape[1],L,kind); return np.array([np.asarray(f(x)) for x in X])
def kernel_from_states(A,B): return np.abs(A.conj()@B.T)**2
def kernel(XA,XB,L,kind='zz'):
    return kernel_from_states(states(XA,L,kind),states(XB,L,kind))
# pairwise "compute-uncompute" circuit (as described in the paper): P(|0..0>) after U(x) U(x')^dagger
_pc={}
def pair_fn(n,L,kind='zz'):
    k=(n,L,kind)
    if k not in _pc:
        dev=qml.device('default.qubit',wires=n)
        @qml.qnode(dev)
        def f(x1,x2):
            fmap(x1,L,kind); qml.adjoint(fmap)(x2,L,kind); return qml.probs(wires=range(n))
        _pc[k]=f
    return _pc[k]
def kernel_pairwise(XA,XB,L,kind='zz',sym=False):
    f=pair_fn(XA.shape[1],L,kind); K=np.zeros((len(XA),len(XB)))
    for i in range(len(XA)):
        for j in range(len(XB)):
            if sym and j<i: K[i,j]=K[j,i]; continue
            K[i,j]=f(XA[i],XB[j])[0]
    return K
# depolarising noise: density-matrix simulation with depolarising channel after every gate
_mc={}
def dm_fn(n,L,p,kind='zz'):
    k=(n,L,p,kind)
    if k not in _mc:
        dev=qml.device('default.mixed',wires=n)
        @qml.qnode(dev)
        def f(x):
            for _ in range(L):
                for i in range(n):
                    qml.Hadamard(wires=i); qml.DepolarizingChannel(p,wires=i); qml.RZ(2*x[i],wires=i); qml.DepolarizingChannel(p,wires=i)
                for i in range(n):
                    j=(i+1)%n
                    qml.CNOT(wires=[i,j]); qml.DepolarizingChannel(p,wires=i); qml.DepolarizingChannel(p,wires=j)
                    qml.RZ(2*(np.pi-x[i])*(np.pi-x[j]),wires=j); qml.DepolarizingChannel(p,wires=j)
                    qml.CNOT(wires=[i,j]); qml.DepolarizingChannel(p,wires=i); qml.DepolarizingChannel(p,wires=j)
            return qml.density_matrix(wires=range(n))
        _mc[k]=f
    return _mc[k]
def kernel_noisy(XA,XB,L,p):
    f=dm_fn(XA.shape[1],L,p); RA=np.array([np.asarray(f(x)) for x in XA]); RB=np.array([np.asarray(f(x)) for x in XB])
    return np.real(np.einsum('aij,bji->ab',RA,RB))
if __name__=='__main__':
    import time
    X=np.random.default_rng(0).uniform(0,np.pi,(6,4))
    for L in (1,2,3):
        K1=kernel(X,X,L); K2=kernel_pairwise(X,X,L); print(L,np.abs(K1-K2).max())
    print('noisy p=0 vs exact', np.abs(kernel_noisy(X,X,2,0.0)-kernel(X,X,2)).max())
