import numpy as np
from pygsp import graphs, filters, plotting
from scipy.linalg import qr
from tqdm import tqdm
from sklearn.linear_model import OrthogonalMatchingPursuit

### CODE FOR GETTING LOCALIZATION MEASURE

def get_locality(wavelets,signal,n_coefs):
    
    # input: wavelets - set of diffusion wavelets
    # s - signal to estimate locality of
    # n_coefs - number of coefficients for OMP 
    
    # output: a number which measures the spread of a signal in a given graph
    
    J = len(wavelets)
    weights = []
    scale = []
    
    print("generating weights")
    for j in range(J):
        n_j = (wavelets[j].shape[1])
        weights += [2**(j)]*n_j
        scale += [j]*n_j
        
    print("Normalizing Wavelets")
    all_wavs = wavelets[0]
    for j in range(1,J):
        all_wavs = np.hstack((all_wavs,wavelets[j]))
        
        
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_coefs, normalize=False)
    
    localization = []
    signal = signal.reshape(len(signal),1)
    
    print("Calculating OMP")
    for j in tqdm(range(signal.shape[1])):
        sig = signal[:,j]
        omp.fit(all_wavs,sig/np.linalg.norm(sig))
        coef = omp.coef_
        localization += [np.sum(coef*weights)]
    
    # plt.bar(range(J),dist)
    # plt.show()
    
    return localization


### CODE FOR GETTING HEAT WAVES 

## MAIN FUNCTION: TAKES INPUT GRAPH G AND OUTPUTS ARRAY OF WAVES AT VARIOUS SCALES

def get_waves(G,epsilon=1e-3,use_reduced=False):
    
    # Input: 
    # G - a pygsp graph 
    # epsilon - tolerance for column reduction
    # use_reduced - dictates whether or not we use a QR decomposition to select a subset 
    # of the columns
    
    # Output: a J x n x ? matrix of diffusion wavelets, where ? depends on whether or not we reduce the number of cols
    
    J = int(np.log(G.N))
    print("Calculating Walk Matrix:")
    T = lazy_walk(G)
    
    print("Calculating Wavelets:")
    if use_reduced:
        return reduced_wavelets(T,J,epsilon)
    else:
        return all_wavelets(T,J)
    
### FIRST OPTION - RETURN ALL DIFFUSED DISTRIBUTIONS AT EVERY SCALE
    
def all_wavelets(T,J):
    
    # Input: An n x n Diffusion operator T, number of scales J
    # Ouput: a J x n x n matrix bank for which bank[j] = T^{2^j}
    
    N = T.shape[0]
    I = np.eye(N)

    bank = [I]
    T_j = T
    for j in tqdm(range(1,J)):
        T_j = np.linalg.matrix_power(T_j,2)
        bank += [T_j]
    return bank

### SECOND OPTION - CUT DOWN ON THE NUMBER OF COLUMNS

def reduced_wavelets(T,J,epsilon):
    
    # Input: An n x n Diffusion operator T, number of scales J
    # Ouput: a J x n x n matrix bank for which bank[j] = T^{2^j}
    
    N = T.shape[0]
    I = np.eye(N)
    bank = [I]
    
    P_j = T
    
    for j in tqdm(range(1,J)):
        print("Calculating Scale " + str(j) + " wavelets:")
        P_j = np.linalg.matrix_power(P_j,2)
        Psi_j_r = column_subset(P_j,epsilon)
        bank += [Psi_j_r]
        print("Used " + str(Psi_j_r.shape[1]) + " columns")
    return bank

### GETTING FEWER COLUMNS 

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

### A FEW HELPER FUNCTIONS

def normalize(A):
    
    # helper function 
    # Input A : an n x m matrix 
    # Output A, but with each column divided by its L2 norm
    
    for i in range(A.shape[1]):
        A[:,i]=A[:,i]/np.linalg.norm(A[:,i])
    return A

def lazy_walk(G):
    
    # Input: pygsp graph G
    # Output: The lazy diffusion operator P = 1/2(I + AD^{-1})
    
    A = G.A
    N = A.shape[0]
    D = [np.sum(row) for row in A]
    Dmin1 = np.diag([1/d for d in D])
    M = A@Dmin1
    P = 1/2 * (np.eye(N) + M)
    return P


## OLD CODE 
def all_wavelets_OLD(T,J):
    
    # Input: An n x n Diffusion operator T, number of scales J
    # Ouput: a J x n x n matrix bank for which bank[j] = T^{2^{j-1}}-T^{2^j}
    
    N = T.shape[0]
    I = np.eye(N)
    bank = np.zeros((J,N,N))
    bank[0] = I.reshape(1,N,N)
    bank[1] = (I-T).reshape(1,N,N)
    T_j = T
    for j in range(2,J):
        T_j_p1 = T_j@T_j
        new_wavs = T_j - T_j_p1
        bank[j] = new_wavs
        T_j = T_j_p1
    return bank


