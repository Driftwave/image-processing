#!/usr/bin/env python3
"""
compute_svd.py - computes image vectors of jpegs

Usage:
compute_svd.py <numpy_data_file> [--standardize]
"""
from docopt import docopt
import numpy as np
from scipy import linalg

MAX_DIMS = 1024


def main():
    args = docopt(__doc__, help=True)
    input_path = args['<numpy_data_file>']
    standardize = args['--standardize']
    assert input_path.endswith('.npy')
    addendum = '-svd_%d' % MAX_DIMS
    if standardize:
        addendum += '_standardized'
    output_path = input_path[:-4] + addendum + '.npy'

    X = np.load(input_path)
    X -= np.mean(X, axis=0)
    if args['--standardize']:
        X /= np.std(X, axis=0)
    cov = np.cov(x, rowvar=False)

    evals, evecs = linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    V = evecs[:, idx[:MAX_DIMS]]
    X = X @ V

    np.save(output_path, X)


if __name__ == '__main__':
    main()
