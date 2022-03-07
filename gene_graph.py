import graphtools
import utils
from DiffusionEMD import DiffusionCheb

def gene_graph(data, graph=None, subsample=None, n_pca=100, n_scales=4, max_scale=12, random_state=None):
    """
    Run DiffusionEMD on data matrix based on multiscale diffusion.
    DiffusionEMD from https://github.com/KrishnaswamyLab/DiffusionEMD.
    
    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    graph: PyGSP graph, optional (default: None)
        Input graph of samples. If not provided, graph built.
    subsample: int or None, optional (default: None)
        If not None, number of samples to keep.
    n_pca : int (default: 100)
        Number of PCs to build graph from data.
    n_scales: int (default: 4)
        Number of scales for DiffusionEMD
    random_state : int, optional (default: None)
        Random seed
        
    Returns
    -------
    embeddings : array-like, shape=[n_features, n_samples * n_scales]
        Outputs multiscale embeddings.
    """
    
    n_cells = data.shape[0]
    n_genes = data.shape[1]
    
    if subsample:
        print ('Density subsampling...')
        data = utils.density_subsample(data,
                                 subsample=subsample,
                                 random_state=random_state)
   
    data = data / data.sum()

    if not graph:
        graph = graphtools.Graph(data,
                     use_pygsp=True,
                     n_pca=n_pca,
                     random_state=random_state,
                     verbose=True)

    print ('Running DiffusionEMD...')
    dc = DiffusionCheb(n_scales=n_scales)
    embeddings = dc.fit_transform(graph.W, data)
    
    return embeddings
