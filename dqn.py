"""Small DQN implemented with NumPy so the experiment has no ML dependency."""
import numpy as np


class Network:
    def __init__(self, seed, width=64, learning_rate=0.001):
        rng = np.random.default_rng(seed)
        self.params = [
            (rng.standard_normal((4, width))*np.sqrt(2/4)).astype(np.float32),
            np.zeros(width, np.float32),
            (rng.standard_normal((width, width))*np.sqrt(2/width)).astype(np.float32),
            np.zeros(width, np.float32),
            (rng.standard_normal((width, 2))*np.sqrt(1/width)).astype(np.float32),
            np.zeros(2, np.float32),
        ]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.lr = learning_rate

    def predict(self, x):
        w1,b1,w2,b2,w3,b3 = self.params
        return np.maximum(0, np.maximum(0, x @ w1 + b1) @ w2 + b2) @ w3 + b3

    def copy_from(self, other):
        for a,b in zip(self.params,other.params): a[:] = b

    def update(self, states, actions, targets):
        w1,b1,w2,b2,w3,b3 = self.params
        z1=states@w1+b1; h1=np.maximum(z1,0)
        z2=h1@w2+b2; h2=np.maximum(z2,0)
        q=h2@w3+b3
        error=q[np.arange(len(actions)),actions]-targets
        grad=np.clip(error,-1,1)/len(actions)  # Huber gradient.
        dq=np.zeros_like(q); dq[np.arange(len(actions)),actions]=grad
        dw3=h2.T@dq; db3=dq.sum(axis=0)
        dz2=(dq@w3.T)*(z2>0)
        dw2=h1.T@dz2; db2=dz2.sum(axis=0)
        dz1=(dz2@w2.T)*(z1>0)
        dw1=states.T@dz1; db1=dz1.sum(axis=0)
        self.t+=1
        for p,g,m,v in zip(self.params,[dw1,db1,dw2,db2,dw3,db3],self.m,self.v):
            np.clip(g,-10,10,out=g)
            m[:]=0.9*m+0.1*g; v[:]=0.999*v+0.001*g*g
            p-=self.lr*(m/(1-0.9**self.t))/(np.sqrt(v/(1-0.999**self.t))+1e-8)

    def save(self,path):
        np.savez(path, **{f'p{i}':p for i,p in enumerate(self.params)})

    def load(self,path):
        with np.load(path) as data:
            for i,p in enumerate(self.params): p[:]=data[f'p{i}']


class Replay:
    def __init__(self, capacity=50000):
        self.s=np.zeros((capacity,4),np.float32)
        self.ns=np.zeros((capacity,4),np.float32)
        self.a=np.zeros(capacity,np.int64)
        self.r=np.zeros(capacity,np.float32)
        self.done=np.zeros(capacity,np.float32)
        self.capacity=capacity; self.count=0

    def add(self,s,a,r,ns,terminated):
        i=self.count%self.capacity
        self.s[i]=s; self.a[i]=a; self.r[i]=r; self.ns[i]=ns
        self.done[i]=terminated  # Time limit remains bootstrappable.
        self.count+=1

    def sample(self,rng,size):
        idx=rng.integers(0,min(self.count,self.capacity),size=size)
        return self.s[idx],self.a[idx],self.r[idx],self.ns[idx],self.done[idx]
