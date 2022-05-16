import numpy as np
from pygsp import graphs, filters, plotting

## usage: (with G a pygsp graph, s a function on the vertex set)
## loc = localizer_emd_approx(G)
## smoothness_measure = loc.smoothness(s)

class localizer_emd_approx:
    
    def __init__(self,G):
        self.G = G
        self.L = G.L.todense()
        self.Linv = np.linalg.pinv(self.L)
        self.T = np.trace(self.L)
       
    def smoothness(self, s):
        s = s/np.sum(s)
        return np.sqrt(self.T) * np.sqrt(s.T@self.Linv@s)
    
    def signal_distance(self, p, q):
        p = p/np.sum(p)
        q = q/np.sum(q)
        delta = p-q
        return np.sqrt(T)*np.sqrt(delta.T@self.Linv@delta)