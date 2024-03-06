import numpy as np
from functools import partial, update_wrapper
from graphviz import Digraph
from sklearn.manifold import Isomap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances

# exports
__all__ = [
    'WrappedPartial', 
    'FunctionNode',
    'flat_brain_isomap',
    'MetricMDS'
]


# Classes for function tree ====================================================

class WrappedPartial:
    """
    A class representing a wrapped partial function.

    Parameters:
    - original_func (callable): The original function to create a partial function from.
    - *args: Positional arguments to fix in the partial function.
    - **kwargs: Keyword arguments to fix in the partial function.
    """

    def __init__(self, original_func, *args, **kwargs):
        """
        Initialize a WrappedPartial instance.

        Parameters:
        - original_func (callable): The original function to create a partial function from.
        - *args: Positional arguments to fix in the partial function.
        - **kwargs: Keyword arguments to fix in the partial function.
        """
        partial_func = partial(original_func, *args, **kwargs)

        # Preserve __doc__ and __name__ attributes
        update_wrapper(partial_func, original_func)
        self.partial_func = partial_func

    def getfunc(self, new_name=None):
        """
        Get the partial function.

        Parameters:
        - new_name (str, optional): If provided, set the __name__ attribute of the partial function to this value.

        Returns:
        - callable: The partial function.
        """
        partial_func = self.partial_func
        if new_name is not None:
            partial_func.__name__ = new_name
        return partial_func 
    
    
class FunctionNode:
    """
    A class representing a node associated with a function.

    Parameters:
    - func (callable): The function associated with the root node.
    - children (list, optional): List of child nodes. Defaults to an empty list.
    """

    def __init__(self, func, children=None):
        """
        Initialize a FunctionTree instance.

        Parameters:
        - func (callable): The function associated with the root node.
        - children (list, optional): List of child nodes. Defaults to an empty list.
        """
        self.func = func
        self.children = children or []

    def add_child(self, child_node):
        """
        Add a child node to the root node.

        Parameters:
        - child_node (FunctionTree): The child node to be added.
        """
        self.children.append(FunctionNode(child_node))

    def find_node(self, target_func):
        """
        Recursively search for a node with a specific function in the tree.

        Parameters:
        - target_func (callable): The target function to search for.

        Returns:
        - FunctionTree or None: The node with the target function, or None if not found.
        """
        if self.func == target_func:
            return self
        else:
            for child in self.children:
                found_node = child.find_node(target_func)
                if found_node:
                    return found_node
        return None

    def add_child_to_node(self, target_func, new_child):
        """
        Add a child node to an arbitrary node in the tree.

        Parameters:
        - target_func (callable): The function associated with the target node.
        - new_child (FunctionTree): The child node to be added to the target node.
        """
        target_node = self.find_node(target_func)
        if target_node:
            target_node.add_child(new_child)
        else:
            print(f"Node with function {target_func} not found.")

    def _evaluate(self, input_value):
        """
        Recursively evaluate the tree and return a list of outputs from all leaf nodes.

        Parameters:
        - input_value: The input value to be used in the function evaluations.

        Returns:
        - list: A list of outputs from all leaf nodes.
        """
        if not self.children:
            if isinstance(input_value, dict):
                return self.func(**input_value),
            return self.func(input_value), 
        else:
            if isinstance(input_value, dict):
                curr_result = self.func(**input_value)
            else:
                curr_result = self.func(input_value)
            child_results = [child.evaluate(curr_result) for child in self.children]
            return  [result for sublist in child_results for result in sublist]
        
    def evaluate(self, input_value):
        """
        Evaluate the tree and return a list of outputs from all leaf nodes.

        Parameters:
        - input_value: The input value to be used in the function evaluations.

        Returns:
        - numpy.ndarray: A numpy array stacking all data produced from the leaf nodes on axis 0.
        The resulting shape is (n_leaves, ...).
        """
        results = self._evaluate(input_value)
        return np.array(results)
        

    def visualize(self, graph=None, parent_name=None, graphviz=None, size=None):
        """
        Generate a graphical representation of the tree using Graphviz.

        Parameters:
        - graph (Digraph, optional): The Graphviz graph. Defaults to None.
        - parent_name (str, optional): The name of the parent node. Defaults to None.
        - graphviz (Digraph, optional): The original Graphviz graph. Defaults to None.
        - size (tuple, optional): The size of the output graph. Defaults to None.

        Returns:
        - Digraph: The Graphviz graph.
        """
        if graph is None:
            graph = Digraph(format='png')
            # Set the size if provided
            if size:
                graph.attr(size=size)
            graphviz = graph
        current_name = str(id(self))
        graph.node(current_name, label=str(self.func.__name__))

        if parent_name is not None:
            graph.edge(parent_name, current_name)

        for i, child in enumerate(self.children):
            child.visualize(graph, current_name, graphviz=graphviz, size=size)

        return graph
    

# Dimensionality reduction for visualization ===================================
    
def flat_brain_isomap(
        coords=None,
        angle=np.pi*1.025,
        horizontal_flip=False,
        vertical_flip=False
    ):
    """
    Applies Isomap dimensionality reduction technique to flatten brain coordinates onto a 2D plane.
    
    Args:
        - coords (array-like): The brain coordinates to be flattened. If unspecified, assumes the coordinates are downloaded
        from https://gist.githubusercontent.com/mwaskom/cb78082d7eede47bed54866fd8cb06b3/raw/9c39627d84c20e65101fff49fe67adb76d6e4155/glasser_coords.txt.
        - angle (float): The angle (in radian) of rotation applied to the flattened coordinates.
        - horizontal_flip (bool, optional): Whether to horizontally flip the flattened coordinates. Default is False.
        - vertical_flip (bool, optional): Whether to vertically flip the flattened coordinates. Default is False.
    
    Returns:
        - array-like: The flattened brain coordinates projected onto a 2D plane.
    """
    isomap = Isomap()
    if coords is None:
        # Load the brain coordinates
        coords = np.loadtxt(
            'https://gist.githubusercontent.com/mwaskom/cb78082d7eede47bed54866fd8cb06b3/raw/9c39627d84c20e65101fff49fe67adb76d6e4155/glasser_coords.txt'
        )
    # Create the 2D rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    if vertical_flip:
        # Create the 2D vertical flip matrix
        vertical_flip_matrix = np.array([
            [1, 0],
            [0, -1]
        ])
    else:
        vertical_flip_matrix = np.eye(2)
    if horizontal_flip:
        # Create the 2D horizontal flip matrix 
        horizontal_flip_matrix = np.array([
            [-1, 0],
            [0, 1]
        ])
    else:
        horizontal_flip_matrix = np.eye(2)
    projection = (
        (isomap.fit_transform(coords) @ rotation_matrix) @ horizontal_flip_matrix
    ) @ vertical_flip_matrix
    return projection


class MetricMDS(BaseEstimator, TransformerMixin):
    """
    Metric Multidimensional Scaling (MDS) implementation.
    """

    def __init__(self, n_components=2):
        """
        Initializes the MetricMDS model.

        Parameters:
        n_components (int): Number of dimensions for embedding. Default is 2.
        """
        self.n_components = n_components
        self.embedding_ = None

    def fit(self, X, y=None):
        """
        Fits the MetricMDS model to the given dissimilarity matrix.

        Parameters:
        X (array-like): The dissimilarity matrix representing pairwise distances.
        y (array-like): Ignored. Present for API consistency.

        Returns:
        self
        """
        # Compute dissimilarity matrix if X is not already a dissimilarity matrix
        if not np.allclose(X, X.T):
            X = pairwise_distances(X, metric='euclidean')

        # Number of samples
        n_samples = X.shape[0]

        # Centering matrix
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples

        # Double centering to get K
        K = -0.5 * np.dot(np.dot(H, X), H)

        # Eigen decomposition of K
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        # Sort eigenvalues and select top components
        sorted_indices = np.argsort(eigenvalues)[::-1][:self.n_components]

        # Diagonal matrix of eigenvalues
        lamb = np.diag(eigenvalues[sorted_indices])

        # Matrix of eigenvectors
        V = eigenvectors[:, sorted_indices]

        # Embedding the data
        self.embedding_ = np.dot(np.sqrt(lamb), V.T).T

        return self

    def fit_transform(self, X, y=None):
        """
        Fits the MetricMDS model to the given dissimilarity matrix and returns the embedded data.

        Parameters:
        X (array-like): The dissimilarity matrix representing pairwise distances.
        y (array-like): Ignored. Present for API consistency.

        Returns:
        np.ndarray: Embedded data in reduced dimensionality.
        """
        self.fit(X)
        return self.embedding_

    