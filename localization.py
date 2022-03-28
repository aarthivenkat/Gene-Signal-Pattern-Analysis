import numpy as np
from pygsp import graphs, filters, plotting
from scipy.linalg import qr
from tqdm import tqdm
from sklearn.linear_model import OrthogonalMatchingPursuit
import phate
from scipy.spatial.distance import pdist, squareform

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
        self.calculated_D = False
        
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
        
        I = np.eye(self.N)
        self.wavelets = [I]
        P_j = np.linalg.matrix_power(self.P,2)
        
        print("Calculating Wavelets Using J = " + str(J))
        
        if use_reduced:
            assert(self.N < 3000)
            Psi_j_tilde = column_subset(I-P_j, epsilon=epslion)
            self.wavelets += [Psi_j_tilde]
            for i in tqdm(range(2,J)):
                P_j_new = np.linalg.matrix_power(P_j,2)
                Psi_j = P_j - P_j_new
                P_j = P_j_new
                self.wavelets += [column_subset(Psi_j,1e-3)]
        else:
            self.wavelets += [I-P_j]
            for i in tqdm(range(2,J)):
                P_j_new = np.linalg.matrix_power(P_j,2)
                Psi_j = P_j - P_j_new
                P_j = P_j_new
                self.wavelets += [Psi_j]

    def randomize_and_reduce(self,epsilon, C):
        self.delta = np.random.randint(0,self.wavelets[0].shape[1], C)
        for j in tqdm(range(self.J)):
            self.wavelets[j] = self.wavelets[j][self.delta][:,self.delta]
            self.wavelets[j] = column_subset(self.wavelets[j],epsilon)
            
    # locality measure
    def GetLocality(self,signal,n_coefs=50):
        
        if self.Precomputed == False:
            print("Flattening and Normalizing Wavelets")
            
            if self.N > 3000:
                print("too many vertices. Clipping to 3000")
                self.randomize_and_reduce(1e-3,3000)
                self.N = 3000
                self.reduced = True
            else:
                self.reduced = False
            
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
        
        if self.reduced:
            signal = signal[self.delta]
            
        signal = signal/np.linalg.norm(signal)
        omp.fit(self.FlatWaves,signal)
        
        print("Compiling Results")
        coef = omp.coef_ * omp.coef_
        return np.sum(coef*self.weights)
      
    def calculate_D(self,method = "phate"):
        if method == "phate":
            phate_op = phate.PHATE(random_state = 0,verbose=False)
            _ = phate_op.fit(self.G)
            t = phate_op._find_optimal_t()
            log_P_to_t = np.log(np.linalg.matrix_power(self.P,t))
            self.D = squareform(pdist(log_P_to_t))
            self.calculated_D = True
        else:
            raise NameError ("Only Supports Phate or Manual Distance at the Moment")
        
    def set_distances(self,D):
        self.D = D
        self.calculated_D = True
        
    def PairDist(self,s):
        s = s.reshape(-1,1)
            
        if self.calculated_D is  False:
            self.calculate_D()
            
        return 1/(np.sum(s))**2 * s.T@self.D@s
            
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
    
    R,P = qr(A,pivoting=True,mode='r')
    A_P = A[:,P]
    
    A_nrm = np.sum(A*A)
    tol = epsilon*A_nrm
    R_nrm = 0
    
    for i in tqdm(range(0,R.shape[0])):
        R_nrm += np.sum(R[i]*R[i])
        err = A_nrm-R_nrm
        if err < tol:
            # print(i,err,R_nrm,A_nrm)
            return A_P[:,:i]
        
    return A_P
