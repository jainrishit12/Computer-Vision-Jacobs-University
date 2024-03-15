import numpy as np

from scipy.linalg import svdvals


def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    Args:
        a: numpy array of shape (x, n)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """

    ### YOUR CODE HERE
    temp = 0
    for i in range(len(b)):
        temp += (a[0][i] * b[i][0])
        # a is a row vector, so we collect entries from columns 0 - 2
        # b is a column vector, so we collect entries rows 0 - 2
    return np.array([[temp]])
    ### END YOUR CODE


def complicated_matrix_function(M, a, b):
    """Implement (a * b) * (M * a.T).

    (optional): Use the `dot_product(a, b)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (1, n).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    out = None
    ### YOUR CODE HERE
    # a.t dot b
    tempDot = dot_product(a, b)
    # M * a.t
    tempMat = np.matmul(M, np.transpose(a))
    # (a.t dot b) * (M * a.t)
    finalVec = tempMat * tempDot

    return finalVec
    ### END YOUR CODE


def svd(M):
    """Implement Singular Value Decomposition.

    (optional): Look up `np.linalg` library online for a list of
    helper functions that you might find useful.

    Args:
        M: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """

    ### YOUR CODE HERE
    return np.linalg.svd(M)
    ### END YOUR CODE


def get_singular_values(M, k):
    """Return top n singular values of matrix.

    (optional): Use the `svd(M)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (m, n).
        k: number of singular values to output.

    Returns:
        singular_values: array of shape (k)
    """
    ### YOUR CODE HERE
    tempArr = np.array([svdvals(M)])
    return tempArr[0][0:k]
    ### END YOUR CODE


def eigen_decomp(M):
    """Implement eigenvalue decomposition.
    
    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """

    ### YOUR CODE HERE
    tempVec, tempMat = np.linalg.eig(M)
    index = np.argsort(tempVec)[::-1]
    tempVec = tempVec[index]
    tempMat = tempMat[:, index]

    ### END YOUR CODE
    return tempVec, tempMat


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """

    ### YOUR CODE HERE
    tempVal, tempVec = np.linalg.eig(M)
    return tempVal[:k], tempVec[:k]
    ### END YOUR CODE
