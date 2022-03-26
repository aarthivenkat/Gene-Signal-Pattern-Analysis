import numpy as np
from pygsp import graphs, filters, plotting
from scipy.linalg import qr
from tqdm import tqdm
from sklearn.linear_model import OrthogonalMatchingPursuit

### say we have a signal s on a pygsp graph G, use would be like so: 

# loc = Localizer(G)
# loc.CalculateWavlets()
# loc.GetLocality(s)

class Localizer:
    
    # initialize with graph and calculate diffusion operator 
    
    def __init__(self,G):
        self.G = G
        self.P = self.DiffusionOperator()
        self.N = G.N
        self.Precomputed = False
        
    # calculate lazy random walk matrix
    def DiffusionOperator(self):
        N = self.G.N
        A = self.G.A
        Dmin1 = np.diag([1/np.sum(row) for row in A])
        P = 1/2 * (np.eye(N)+A@Dmin1)
        return P
    
    # calculate wavelets 
    def CalculateWavelets(self,use_reduced=False,J=-1,epsilon=1e-3):
        
        # assert(self.P)
        
        if J == -1:
            J = int(np.log(self.N))
        self.J = J
        
        self.wavelets = [np.eye(self.N)]
        P_j = self.P
        
        print("Calculating Wavelets Using J = " + str(J))
        for i in tqdm(range(1,J)):
            
            P_j = np.linalg.matrix_power(P_j,2)
            if use_reduced:
                Psi_j = column_subset(P_j,epsilon)
            else:
                Psi_j = P_j
            
            self.wavelets += [Psi_j]
    
    # locality measure
    def GetLocality(self,signal):
        
        if self.Precomputed == False:
            print("Flattening and Normalizing Wavelets")
            
            ncols = [self.wavelets[i].shape[1] for i in range(self.J)]
            TOT = np.sum(ncols)
            
            self.FlatWaves = np.zeros((self.N,TOT))
            self.weights = []
            curr = 0
            
            for j in tqdm(range(self.J)):
                normalized_j = normalize(self.wavelets[j])
                last = curr + ncols[j]
                self.FlatWaves[:,curr:last] = normalized_j
                self.weights += [2**(j)]*self.wavelets[j].shape[1]
                curr = last
            self.Precomputed = True
            
        print("Calculating Matching Pursuit")
        omp = OrthogonalMatchingPursuit(tol = 1e-3, normalize=True)
        signal = signal/np.linalg.norm(signal)
        omp.fit(self.FlatWaves,signal)
        
        print("Compiling Results")
        return np.sum(np.abs(omp.coef_)*self.weights)
            
def normalize(A):
    
    # helper function 
    # Input A : an n x m matrix 
    # Output A, but with each column divided by its L2 norm
    
    for i in range(A.shape[1]):
        A[:,i]=A[:,i]/np.linalg.norm(A[:,i])
    return A

def column_subset(A,epsilon):
    
    # Input: an m x n matrix A, tolerance epsilon
    # Output: A subset of A's columns s.t. the projection of A into these columns 
    # can approximate A with error < epsilon |A|_2
    
    Q,R,P = qr(A,pivoting=True)
    A_P = A[:,P]
    
    A_nrm = np.sum(A*A)
    tol = epsilon*A_nrm
    R_nrm = 0
    
    for i in tqdm(range(0,A.shape[1])):
        R_nrm += np.sum(R[i]*R[i])
        err = A_nrm-R_nrm
        if err < tol:
            # print(i,err,R_nrm,A_nrm)
            return A_P[:,:i]
        
    return A_P
        
