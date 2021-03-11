from scipy.sparse import csr_matrix
import numpy as np

def sparse_matlab(i, j, v, m, n):
    """
    Create and compressing a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values 
            Size n1
        j: 1-D array representing the index 2 values 
            Size n1
        v: 1-D array representing the values 
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return csr_matrix((v, (i, j)), shape=(m, n))


def pad_array(arr: list, fill_value=0):
    '''
    Fill zero for variable length 2D data
    '''

    # Get lengths of each row of data
    lens = np.array([len(a) for a in arr])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    if fill_value:
        out = np.full_like(mask, fill_value=fill_value, dtype=type(arr[0]))
    else: # fill with zero
        out = np.zeros_like(mask, dtype=type(arr[0]))
    out[mask] = np.concatenate(arr)

    #### another implementation
    # max_len = np.max([len(a) for a in arr])
    # out= [np.pad(a, (0, max_len - len(a)), 
                # 'constant', constant_values=fill_value) for a in arr]
    # out = np.asarray(out)
    return out

def setdiff2d(arr1, arr2):
    '''
    setdiff: similar to matlab function along axis 0 for
        - arr1: a 2D array
        - arr2: a 2D array
    '''
    assert len(arr1) == len(arr2)

    diff = []
    for a1, a2 in zip(arr1, arr2):
        a = set(a1).difference(a2)
        diff.append(list(a))
    diff = pad_array(diff)
    return diff