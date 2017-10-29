#!/usr/bin/env python
# encoding: utf-8

r"""
Shallow water flow
==================

Solve the one-dimensional shallow water equations including bathymetry:

.. math::
    h_t + (hu)_x & = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x & = -g h b_x.

Here h is the depth, u is the velocity, g is the gravitational constant, and b
the bathymetry.  
"""

from __future__ import absolute_import

import numpy
import matplotlib
import matplotlib.pyplot as plt


def deposition(state, n):
    """Deposition function for species *n*

    """
    c = numpy.where(state.q[0, :] > state.problem_data['dry_tolerance'],
                    state.q[2 + n, :] / state.q[0, :],
                    0.0)
    omega_0 = state.problem_data['omega_0'][n]
    m = 2

    return alpha(state, n) * c * omega_0 * (1.0 - alpha(state, n) * c)**m


def erosion(state, n):
    """Erosion function for species *n*

    """

    phi = state.problem_data['phi']
    g = state.problem_data['grav']
    dry_tol = state.problem_data['dry_tolerance']
    mannings_n = state.problem_data['mannings_n']
    s = state.problem_data['s'][n]
    d = state.problem_data['d'][n]
    theta_c = state.problem_data['theta_c'][n]

    u = numpy.where(state.q[0, :] > dry_tol, 
                    state.q[1, :] / state.q[0, :],
                    numpy.zeros(state.q.shape[1]))
    u_star = state.q[0, :] * numpy.sqrt(g * state.q[0, :]) * numpy.abs(mannings_n**2 * u**2 / state.q[0, :]**(4.0 / 3.0)) 
    theta = u_star**2 / (s * g * d)

    return numpy.where(theta >= theta_c, 
                       phi * (theta - theta_c) * u / (state.q[0, :] * d**(0.2)),
                       0.0)

def rho_mix(state, n):
    c = numpy.where(state.q[0, :] > state.problem_data['dry_tolerance'],
                    state.q[2 + n, :] / state.q[0, :],
                    0.0)
    return state.problem_data['rho_fluid'] * (1.0 - c) + state.problem_data['rho'][n] * c


def alpha(state, n):

    # NOTE:  Maybe we want to have another tolerance for this
    # This set "over_c" to 3.0 as the eventual minimization should lead to the
    # same value
    over_c = numpy.where(state.q[2 + n, :] > state.problem_data['dry_tolerance'],
                        (1.0 - state.problem_data['p']) * state.q[0, :] / state.q[2 + n, :],
                         10.0)

    return numpy.fmin(numpy.ones(over_c.shape) * 2.0, over_c)


def source_term(solver, state, dt):

    g = state.problem_data['grav']
    mannings_n = state.problem_data['mannings_n']
    dry_tol = state.problem_data['dry_tolerance']
    rho_fluid = state.problem_data['rho_fluid']
    rho = state.problem_data['rho']
    rho_0 = state.problem_data['rho_0']
    p = state.problem_data['p']

    E = numpy.empty((state.q.shape[1], state.problem_data['num_species']))
    D = numpy.empty((state.q.shape[1], state.problem_data['num_species']))
    for n in range(state.problem_data['num_species']):
        E[:, n] = erosion(state, n)
        D[:, n] = deposition(state, n)
    
    # Friction
    u = numpy.where(state.q[0, :] > dry_tol, 
                    state.q[1, :] / state.q[0, :], 
                    numpy.zeros(state.q[0, :].shape[0]))
    gamma = u * g * mannings_n**2 / state.q[0, :]**(4.0 / 3.0)
    state.q[1, :] /= (1.0 + dt * gamma)

    # Update to mass
    for n in range(state.problem_data['num_species']):
        state.q[0, :] += dt * numpy.sum(E[:, n] - D[:, n]) / (1.0 - p)

    # Update to momentum
    dx = state.patch.dimensions[0].delta
    dcdx = numpy.zeros(state.q.shape[1])
    term_1 = 0.0
    term_2 = 0.0
    for n in range(state.problem_data['num_species']):
        c = numpy.where(state.q[0, :] > dry_tol,
                        state.q[2 + n, :] / state.q[0, :],
                        numpy.zeros(state.q.shape[1]))
        u = numpy.where(state.q[0, :] > dry_tol, 
                        state.q[1, :] / state.q[0, :], 
                        numpy.zeros(state.q[0, :].shape[0]))    
        
        # Compute derivative
        dcdx[0] = (c[1] - c[0]) / dx
        dcdx[1:-1] = (c[2:] - c[:-2]) / (2.0 * dx)
        dcdx[-1] = (c[-1] - c[-2]) / dx
    
        term_1 += -(rho[n] - rho_fluid) * g * state.q[0, :]**2 / (2.0 * rho_mix(state, n)) * dcdx
        term_2 += -(rho_0[n] - rho_mix(state, n)) * (E[:, n] - D[:, n]) * u / (rho_mix(state, n) * (1.0 - p))

    state.q[1, :] += dt * (term_1 + term_2)

    # Update to sediment mass
    for n in range(state.problem_data['num_species']):
        state.q[2 + n, :] += dt * (E[:, n] - D[:, n])

    # Update to bathymetry
    for n in range(state.problem_data['num_species']):
        state.aux[0, :] += dt * (D[:, n] - E[:, n]) / (1.0 - p)


def setup(kernel_language='Fortran', solver_type='classic', use_petsc=False,
          outdir='./_output'):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    import shallow_sediment_1d
    solver = pyclaw.ClawSolver1D(shallow_sediment_1d.shallow_fwave_1d)
    solver.kernel_language = 'Python'
    solver.limiters = pyclaw.limiters.tvd.vanleer
    solver.fwave = True
    solver.num_waves = 3
    solver.num_eqn = 3
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.source_split = 1
    solver.step_source = source_term

    num_species = 1
    xlower = 0.0
    xupper = 50e3
    x = pyclaw.Dimension(xlower, xupper, 1000, name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, 2 + num_species, 1)

    # Problem data
    state.problem_data['grav'] = 9.81
    state.problem_data['dry_tolerance'] = 1e-3
    state.problem_data['mannings_n'] = 0.03
    state.problem_data['nu'] = 1.2e-6
    state.problem_data['rho_fluid'] = 1000.0
    
    state.problem_data['num_species'] = num_species
    state.problem_data['phi'] = 0.015
    state.problem_data['p'] = 0.4
    
    state.problem_data['rho'] = numpy.array([2000.0])
    state.problem_data['d'] = numpy.array([0.004])
    state.problem_data['theta_c'] = numpy.array([0.045])
    
    state.problem_data['s'] = numpy.empty(num_species)
    state.problem_data['rho_0'] = numpy.empty(num_species)
    state.problem_data['omega_0'] = numpy.empty(num_species)
    for n in range(state.problem_data['num_species']):
        state.problem_data['s'][n] = state.problem_data['rho'][n] / state.problem_data['rho_fluid'] - 1.0
        state.problem_data['rho_0'][n] = state.problem_data['rho_fluid'] + state.problem_data['rho'][n] * (1.0 - state.problem_data['p'])
        state.problem_data['omega_0'][n] =   numpy.sqrt((13.95 * state.problem_data['nu'] / state.problem_data['d'][n])**2 \
                                           + 1.09 * state.problem_data['s'][n] * state.problem_data['grav'] * state.problem_data['d'][n])  \
                                           - 13.95 * state.problem_data['nu'] / state.problem_data['d'][n]

    xc = state.grid.x.centers
    state.aux[0, :] = 0.0
    state.q[0, :] = 40.0 * numpy.ones(x.centers.shape[0]) * (x.centers <= 25e3) \
                  + 2.0 * numpy.ones(x.centers.shape[0]) * (x.centers > 25e3)
    state.q[1, :] = 0.0
    state.q[2:, :] = 0.0

    claw = pyclaw.Controller()
    claw.output_style = 1
    claw.keep_copy = False
    claw.num_output_times = 10
    claw.tfinal = 60.0 * 20
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.setplot = setplot
    claw.write_aux_always = True

    if outdir is not None:
        claw.outdir = outdir
    else:
        claw.output_format = None


    return claw


#--------------------------
def setplot(plotdata):
#--------------------------
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """ 
    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Plot variables
    def bathy(current_data):
        return current_data.aux[0, :]

    def eta(current_data):
        return current_data.q[0, :] + bathy(current_data)

    def velocity(current_data, dry_tol=1e-3):
        return numpy.where(current_data.q[0, :] >= dry_tol,
                           current_data.q[1, :] / current_data.q[0, :],
                           numpy.zeros(current_data.q.shape[1]))

    def concentration(current_data, dry_tol=1e-3):
        return numpy.where(current_data.q[0, :] >= dry_tol,
                           current_data.q[2, :] / current_data.q[0, :],
                           numpy.zeros(current_data.q.shape[1]))

    rgb_converter = lambda triple: [float(rgb) / 255.0 for rgb in triple]

    xlimits = [23000, 27000]
    xlimits = [0.0, 50e3]

    # Figure for depth
    plotfigure = plotdata.new_plotfigure(name='Depth')

    # Axes for water depth
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = [-10.0, 40.0]
    plotaxes.title = 'Water Depth'

    plotitem = plotaxes.new_plotitem(plot_type='1d_fill_between')
    plotitem.plot_var = eta
    plotitem.plot_var2 = bathy
    plotitem.color = rgb_converter((67,183,219))

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = bathy
    plotitem.color = 'k'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = eta
    plotitem.color = 'k'

    # Figure for Velocity
    plotfigure = plotdata.new_plotfigure(name='Velocity')

    # Axes for velocity
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = xlimits
    plotaxes.title = 'Velocity'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = velocity
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}

    # Figure for concentration
    plotfigure = plotdata.new_plotfigure(name='Concentration')

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = xlimits
    plotaxes.title = 'Concentration'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = concentration
    plotitem.color = 'k'
    plotitem.kwargs = {'linewidth':3}

    # ================
    #  Combined Figre
    # ================
    plotfigure = plotdata.new_plotfigure(name='Combined')
    default_figsize = [6.4, 4.8]
    default_figsize = matplotlib.rcParams['figure.figsize']
    plotfigure.kwargs['figsize'] = (6.4, 6 * 1.3)

    # Depth + sediment
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = "subplot(3, 1, 1)"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = [-10.0, 40.0]
    plotaxes.title = 'Water Depth'

    plotitem = plotaxes.new_plotitem(plot_type='1d_fill_between')
    plotitem.plot_var = eta
    plotitem.plot_var2 = bathy
    plotitem.color = rgb_converter((67,183,219))

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = bathy
    plotitem.color = 'k'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = eta
    plotitem.color = 'k'

    # Velocity
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = "subplot(3, 1, 2)"
    plotaxes.xlimits = xlimits
    plotaxes.title = 'Velocity'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = velocity
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}

    # Concentration
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = xlimits
    plotaxes.title = 'Concentration'
    plotaxes.axescmd = "subplot(3, 1, 3)"

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = concentration
    plotitem.color = 'k'
    plotitem.kwargs = {'linewidth':3}

    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
