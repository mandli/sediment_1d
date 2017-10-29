#!/usr/bin/env python
# encoding: utf-8
r"""
Riemann solvers for the shallow water equations.

The available solvers are:
 * Roe - Use Roe averages to caluclate the solution to the Riemann problem
 * HLL - Use a HLL solver
 * Exact - Use a newton iteration to calculate the exact solution to the
        Riemann problem

.. math::
    q_t + f(q)_x = 0

where

.. math::
    q(x,t) = \left [ \begin{array}{c} h \\ h u \end{array} \right ],

the flux function is

.. math::
    f(q) = \left [ \begin{array}{c} h u \\ hu^2 + 1/2 g h^2 \end{array}\right ].

and :math:`h` is the water column height, :math:`u` the velocity and :math:`g`
is the gravitational acceleration.
"""

from __future__ import absolute_import
import numpy as np
from six.moves import range

num_eqn = 3
num_waves = 3

def shallow_fwave_1d(q_l, q_r, aux_l, aux_r, problem_data):
    r"""Shallow water Riemann solver using fwaves
    """

    g = problem_data['grav']
    dry_tolerance = problem_data['dry_tolerance']
    num_species = problem_data['num_species']

    num_rp = q_l.shape[1]
    num_eqn = 2 + num_species
    num_waves = 3

    # Output arrays
    fwave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )

    # Extract state
    u_l = np.where(q_l[0, :] > dry_tolerance,
                   q_l[1, :] / q_l[0, :], 0.0)
    u_r = np.where(q_r[0, :] > dry_tolerance,
                   q_r[1, :] / q_r[0, :], 0.0)
    phi_l = q_l[0, :] * u_l**2 + 0.5 * g * q_l[0, :]**2
    phi_r = q_r[0, :] * u_r**2 + 0.5 * g * q_r[0, :]**2
    h_bar = 0.5 * (q_l[0, :] + q_r[0, :])

    # Speeds
    u_hat = (np.sqrt(g * q_l[0, :]) * u_l + np.sqrt(g * q_r[0, :]) * u_r)      \
            / (np.sqrt(g * q_l[0, :]) + np.sqrt(g * q_r[0, :]))
    c_hat = np.sqrt(g * h_bar)
    s[0, :] = np.amin(np.vstack((u_l - np.sqrt(g * q_l[0, :]),
                                 u_hat - c_hat)), axis=0)
    s[1, :] = np.amax(np.vstack((u_r + np.sqrt(g * q_r[0, :]),
                                 u_hat + c_hat)), axis=0)
    s[2, :] = u_hat

    delta1 = q_r[1, :] - q_l[1, :]
    delta2 = phi_r - phi_l + g * h_bar * (aux_r[0, :] - aux_l[0, :])

    beta1 = (s[1, :] * delta1 - delta2) / (s[1, :] - s[0, :])
    beta2 = (delta2 - s[0, :] * delta1) / (s[1, :] - s[0, :])

    fwave[0, 0, :] = beta1
    fwave[1, 0, :] = beta1 * s[0, :]
    fwave[2:, 0, :] = 0.0

    fwave[0, 1, :] = beta2
    fwave[1, 1, :] = beta2 * s[1, :]
    fwave[2:, 1, :] = 0.0
    
    fwave[0, 2:, :] = 0.0
    fwave[1, 2:, :] = 0.0
    for n in range(num_species):
        fwave[2 + n, 2, :] = q_r[2 + n, :] - q_l[2 + n, :]

    for m in range(num_eqn):
        for mw in range(num_waves):
            amdq[m, :] += (s[mw, :] < 0.0) * fwave[m, mw, :]
            apdq[m, :] += (s[mw, :] > 0.0) * fwave[m, mw, :]

            amdq[m, :] += (s[mw, :] == 0.0) * fwave[m, mw, :] * 0.5
            apdq[m, :] += (s[mw, :] == 0.0) * fwave[m, mw, :] * 0.5

    return fwave, s, amdq, apdq


