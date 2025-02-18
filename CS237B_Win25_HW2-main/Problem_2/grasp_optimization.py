#!/usr/bin/env python

import cvxpy as cp
import numpy as np
import pdb  

from utils import *

def solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False):
    """
    Solves an SOCP of the form:

    minimize(h^T x)
    subject to:
        ||A_i x + b_i||_2 <= c_i^T x + d_i    for all i
        F x == g

    Args:
        x       - cvx variable.
        As      - list of A_i numpy matrices.
        bs      - list of b_i numpy vectors.
        cs      - list of c_i numpy vectors.
        ds      - list of d_i numpy vectors.
        F       - numpy matrix.
        g       - numpy vector.
        h       - numpy vector.
        verbose - whether to print verbose cvx output.

    Return:
        x - the optimal value as a numpy array, or None if the problem is
            infeasible or unbounded.
    """
    objective = cp.Minimize(h.T @ x)
    constraints = []
    for A, b, c, d in zip(As, bs, cs, ds):
        constraints.append(cp.SOC(c.T @ x + d, A @ x + b))
    constraints.append(F @ x == g)
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    if prob.status in ['infeasible', 'unbounded']:
        return None

    return x.value

def grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext):
    """
    Solve the grasp force optimization problem as an SOCP. Handles 2D and 3D cases.

    Args:
        grasp_normals   - list of M surface normals at the contact points, pointing inwards.
        points          - list of M grasp points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).
        wrench_ext      - external wrench applied to the object.

    Return:
        f
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)
    transformations = [compute_local_transformation(n) for n in grasp_normals]

    ########## Your code starts here ##########
    As = []
    bs = []
    cs = []
    ds = []

    x = cp.Variable(M*D + 1)

    F = np.zeros((N, M*D + 1))

    for i in range(M):
        if D == 2:
            # friction cone constraints
            A_i = np.zeros((1, M*D + 1))
            A_i[0, i*D] = 1
            As.append(A_i)
            bs.append(0)
            c_i = np.zeros(M*D + 1)
            c_i[i*D + 1] = friction_coeffs[i]
            cs.append(c_i)
            ds.append(0)
            # force balance constraints
            A_j = np.zeros((2, M*D + 1))
            A_j[0, i*D] = 1
            A_j[1, i*D + 1] = 1
            As.append(A_j)
            bs.append(0)
            c_j = np.zeros(M*D + 1)
            c_j[-1] = 1
            cs.append(c_j)
            ds.append(0)
            # transformation matrix
            F[0:D, i*D:(i + 1) * D] = transformations[i]
            F[D:, i*D:(i + 1) * D] = np.cross(points[i],transformations[i])

        if D == 3:
            # friction cone constraints
            A_i = np.zeros((2, M*D + 1))
            A_i[0, i*D] = 1
            A_i[1, i*D + 1] = 1
            As.append(A_i)
            bs.append(0)
            c_i = np.zeros(M*D + 1)
            c_i[i*D + 2] = friction_coeffs[i]
            cs.append(c_i)
            ds.append(0)
            # force balance constraints
            A_j = np.zeros((3, M*D + 1))
            A_j[0, i*D] = 1
            A_j[1, i*D + 1] = 1
            A_j[2, i*D + 2] = 1
            As.append(A_j)
            bs.append(0)
            c_j = np.zeros(M*D + 1)
            c_j[-1] = 1
            cs.append(c_j)
            ds.append(0)
            # transformation matrix
            F[0:D, i*D:(i + 1) * D] = transformations[i]
            F[D:, i*D:(i + 1) * D] = np.cross(points[i],transformations[i].T)
        

    g = -1 * wrench_ext
    h = np.zeros(M*D + 1)
    h[-1] = 1

    solution = solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False)

    # TODO: extract the grasp forces from x as a stacked 1D vector

    if solution is None:
        return None
    
    f = solution[:-1]

    ########## Your code ends here ##########

    # Transform the forces to the global frame
    F = f.reshape(M,D)
    forces = [T.dot(f) for T, f in zip(transformations, F)]

    return forces

def precompute_force_closure(grasp_normals, points, friction_coeffs):
    """
    Precompute the force optimization problem so that force closure grasps can
    be found for any arbitrary external wrench without redoing the optimization.

    Args:
        grasp_normals   - list of M contact normals, pointing inwards from the object surface.
        points          - list of M contact points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).

    Return:
        force_closure(wrench_ext) - a function that takes as input an external wrench and
                                    returns a set of forces that maintains force closure.
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)

    ########## Your code starts here ##########
    # TODO: Precompute the optimal forces for the 12 signed unit external
    #       wrenches and store them as rows in the matrix F. This matrix will be
    #       captured by the returned force_closure() function.

    F = np.zeros((2*N, M*D))

    for i in range(N):
        pos_wrench = np.zeros(N)
        pos_wrench[i] = 1
        forces = grasp_optimization(grasp_normals, points, friction_coeffs, pos_wrench)
        F[2 * i] = np.concatenate([f for f in forces])

        neg_wrench = np.zeros(N)
        neg_wrench[i] = -1
        forces = grasp_optimization(grasp_normals, points, friction_coeffs, neg_wrench)
        F[2 * i + 1] = np.concatenate([f for f in forces])

    ########## Your code ends here ##########

    def force_closure(wrench_ext):
        """
        Return a set of forces that maintain force closure for the given
        external wrench using the precomputed parameters.

        Args:
            wrench_ext - external wrench applied to the object.

        Return:
            forces - grasp forces as a list of M numpy arrays.
        """

        ########## Your code starts here ##########
        # TODO: Compute the force closure forces as a stacked vector of shape (M*D)
        f = np.zeros(M*D)

        wrench_pos = np.maximum(wrench_ext, 0)
        wrench_neg = np.maximum(-wrench_ext, 0)

        for i in range(N):
            f += F[2 * i] * wrench_pos[i] + F[2 * i + 1] * wrench_neg[i]
        ########## Your code ends here ##########

        forces = [f_i for f_i in f.reshape(M,D)]
        return forces

    return force_closure
