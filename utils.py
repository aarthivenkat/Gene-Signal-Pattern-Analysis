import numpy as np
import scipy
import sklearn
import pandas as pd
import scprep
import pygsp

def heat_filter(data, graph, tau=60, chebyshev_order=32):
    """
    Smooth gene signals on graph
    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    graph : PyGSP graph 
        Input cell-cell graph
    tau : Heat filter hyperparameter, optional (default: 60)
        Scale of heat diffusion
    chebyshev_order: Chebyshev order, optional (default: 32)
        Order of Chebyshev polynomial approximation

    Returns
    -------
    data_diffused: array_like, shape=[n_samples, n_features]
        Diffused output data
    """
    
    data = data / data.sum(axis=0) # create probability distribution
    graph.estimate_lmax()
    f = pygsp.filters.Heat(graph, tau=tau)
    data_diffused = f.filter(data, method="chebyshev", order=chebyshev_order)
    
    return data_diffused

def density_subsample(data, subsample=2000, n_neighbors=3, random_state=None):
    """
    Density subsample data matrix. 
    Solution from https://github.com/KrishnaswamyLab/scprep/issues/106
    
    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    subsample : int (default: 2000)
        Number of samples to keep
    n_neighbors: int (default: 3)
        Number of neighbors to use for k-neighbors queries
    random_state : int, optional (default: None)
        Random seed
        
    Returns
    -------
    data_subsample : array-like, shape=[subsample, n_features]
        Subsetted output data
    """
    
    n_cells = data.shape[0]  
    if isinstance(data, pd.core.frame.DataFrame):
        indata = scipy.sparse.csr_matrix(data.values)
    elif isinstance(data, np.ndarray):
        indata = data
    else:
        raise ValueError(f'Expected data input to be array-like, shape=[n_samples, n_features]')
        
    if n_cells < subsample:
        raise ValueError(f'Expected subsample ({subsample}) <= n_cells ({n_cells})')
    
    np.random.seed(random_state)
    distances, _ = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors).fit(indata).kneighbors()
    distances = distances.max(axis=1)
    p = distances / distances.sum()
    data_subsample = scprep.select.select_rows(data, idx=np.random.choice(n_cells, subsample, p=p, replace=False))
    
    return data_subsample