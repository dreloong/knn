import numpy as np

def reduce_dim(examples, explained_var_ratio=0.85, debug=False):
    X = examples - np.mean(examples, axis=0)
    eig_vals, eig_vecs = np.linalg.eig(np.cov(X.T))

    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[idx]

    var_sum = np.sum(eig_vals)
    var_cumsum = 0
    n_components = 0
    for eig_val in eig_vals:
        var_cumsum += eig_val
        n_components += 1
        if var_cumsum >= var_sum * explained_var_ratio:
            if debug:
                print '{} components selected.'.format(n_components)
                print '{:.2%} variance explained.'.format(var_cumsum / var_sum)
            break

    W = eig_vecs[:n_components].T
    return np.dot(X, W)
