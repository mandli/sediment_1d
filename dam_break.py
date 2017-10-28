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


def deposition(state): 
    c = numpy.where(state.q[0, :] > state.problem_data['dry_tolerance'],
                    state.q[2, :] / state.q[0, :],
                    numpy.zeros(state.q.shape[1]))
    omega_0 = state.problem_data['omega_0']
    m = 2
    
    # print("alpha: ", alpha(state))
    # import pdb; pdb.set_trace()

    return alpha(state) * c * omega_0 * (1.0 - alpha(state) * c)**m


# def erosion(state):
#     s = state.problem_data['s']
#     d = state.problem_data['d']
#     nu = state.problem_data['nu']
#     g = state.problem_data['grav']
#     theta_c = state.problem_data['theta_c']
#     p = state.problem_data['p']

#     u = numpy.where(state.q[0, :] > state.problem_data['dry_tolerance'], 
#                     state.q[1, :] / state.q[0, :],
#                     numpy.zeros(state.q.shape[1]))
#     u_star = state.q[0, :] * numpy.sqrt(state.problem_data['grav'] * state.q[0, :]) * numpy.abs(state.problem_data['mannings_n']**2 * u**2 / state.q[0, :]**(4.0 / 3.0)) 
#     theta = u_star**2 / (state.problem_data['s'] * state.problem_data['grav'] * state.problem_data['d'])
#     R = numpy.sqrt(s * g * d) * d / nu
#     U_inf = 7.0 * u / 6.0

#     return numpy.where(theta >= 160.0 / R**(0.8) * (1.0 - p) / theta_c * (theta - theta_c) * d * U_inf / state.q[0, :]

def erosion(state):
    phi = state.problem_data['phi']
    u = numpy.where(state.q[0, :] > state.problem_data['dry_tolerance'], 
                    state.q[1, :] / state.q[0, :],
                    numpy.zeros(state.q.shape[1]))
    u_star = state.q[0, :] * numpy.sqrt(state.problem_data['grav'] * state.q[0, :]) * numpy.abs(state.problem_data['mannings_n']**2 * u**2 / state.q[0, :]**(4.0 / 3.0)) 
    theta = u_star**2 / (state.problem_data['s'] * state.problem_data['grav'] * state.problem_data['d'])
    
    # print("u_star: ", u_star)
    # print("theta_c: ", state.problem_data['theta_c'])
    # print("Theta: ", theta)
    # import pdb; pdb.set_trace()

    return numpy.where(theta >= state.problem_data['theta_c'], 
                       phi * (theta - state.problem_data['theta_c']) * u / (state.q[0, :] * state.problem_data['d']**(0.2)),
                       numpy.zeros(state.q.shape[1]))

def rho_mix(state):
    c = numpy.where(state.q[0, :] > state.problem_data['dry_tolerance'],
                    state.q[2, :] / state.q[0, :],
                    numpy.zeros(state.q.shape[1]))
    return state.problem_data['rho_w'] * (1.0 - c) + state.problem_data['rho_s'] * c


def alpha(state):
    over_c = numpy.where(state.q[0, :] > state.problem_data['dry_tolerance'],
                    (1.0 - state.problem_data['p']) / (state.q[2, :] / state.q[0, :]),
                    numpy.infty)

    return numpy.fmin(numpy.ones(over_c.shape[0]) * 2.0, over_c)


def before_step(solver, state):
    pass


def source_term(solver, state, dt):

    g = state.problem_data['grav']
    mannings_n = state.problem_data['mannings_n']
    dry_tol = state.problem_data['dry_tolerance']
    rho_w = state.problem_data['rho_w']
    rho_s = state.problem_data['rho_s']
    rho_0 = state.problem_data['rho_0']
    p = state.problem_data['p']

    if numpy.any(erosion(state) < 0.0):
        raise ValueError("Erosion < 0.0")

    if numpy.any(deposition(state) < 0.0):
        raise ValueError("Deposition < 0.0")
    
    # Friction
    if mannings_n > 0.0:
        u = numpy.where(state.q[0, :] > dry_tol, 
                        state.q[1, :] / state.q[0, :], 
                        numpy.zeros(state.q[0, :].shape[0]))
        gamma = u * g * mannings_n**2 / state.q[0, :]**(4.0 / 3.0)
        state.q[1, :] /= (1.0 + dt * gamma)

    # Update to mass
    state.q[0, :] += dt * (erosion(state) - deposition(state)) / (1.0 - state.problem_data['p'])

    # Update to momentum
    c = numpy.where(state.q[0, :] > dry_tol,
                    state.q[2, :] / state.q[0, :],
                    numpy.zeros(state.q.shape[1]))
    u = numpy.where(state.q[0, :] > dry_tol, 
                    state.q[1, :] / state.q[0, :], 
                    numpy.zeros(state.q[0, :].shape[0]))
    
    dx = state.patch.dimensions[0].delta
    dcdx = numpy.zeros(c.shape[0])
    dcdx[0] = (c[1] - c[0]) / dx
    dcdx[1:-1] = (c[2:] - c[:-2]) / (2.0 * dx)
    dcdx[-1] = (c[-1] - c[-2]) / dx

    # print("Erosion: ")
    # print(erosion(state))
    # print("Deposition: ")
    # print(deposition(state))
    # import pdb; pdb.set_trace()
    
    state.q[1, :] += -dt * (rho_0 - rho_mix(state)) * (erosion(state) - deposition(state)) * u / (rho_mix(state) * (1.0 - p))
    state.q[1, :] += -dt * (rho_s - rho_w) * g * state.q[0, :]**2 / (2.0 * rho_mix(state)) * dcdx

    # Update to sediment mass
    state.q[2, :] += dt * (erosion(state) - deposition(state))

    # Update to bathymetry
    state.aux[0, :] += dt * (deposition(state) - erosion(state)) / (1.0 - p)


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
    solver.before_step = before_step
    solver.source_split = 1
    solver.step_source = source_term

    xlower = 0.0
    xupper = 50e3
    x = pyclaw.Dimension(xlower, xupper, 10000, name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, 3, 1)

    # Gravitational constant
    state.problem_data['grav'] = 9.81
    state.problem_data['dry_tolerance'] = 1e-3
    state.problem_data['sea_level'] = 0.0
    state.problem_data['mannings_n'] = 0.03
    state.problem_data['rho_s'] = 2000.0
    state.problem_data['rho_w'] = 1000.0
    state.problem_data['d'] = 0.004
    state.problem_data['phi'] = 0.015
    state.problem_data['nu'] = 1.2e-6
    state.problem_data['s'] = state.problem_data['rho_s'] / state.problem_data['rho_w'] - 1.0
    state.problem_data['p'] = 0.4
    state.problem_data['rho_0'] = state.problem_data['rho_w'] + state.problem_data['rho_s'] * (1.0 - state.problem_data['p'])
    state.problem_data['omega_0'] =   numpy.sqrt((13.95 * state.problem_data['nu'] / state.problem_data['d'])**2 \
                                    + 1.09 * state.problem_data['s'] * state.problem_data['grav'] * state.problem_data['d'])  \
                                    - 13.95 * state.problem_data['nu'] / state.problem_data['d']
    state.problem_data['theta_c'] = 0.045

    xc = state.grid.x.centers
    state.aux[0, :] = 0.0
    state.q[0, :] = 40.0 * numpy.ones(x.centers.shape[0]) * (x.centers <= 25e3) \
                  + 2.0 * numpy.ones(x.centers.shape[0]) * (x.centers > 25e3)
    state.q[1, :] = 0.0
    state.q[2, :] = 0.0

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
    plotfigure = plotdata.new_plotfigure(name='Depth', figno=0)

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
    plotfigure = plotdata.new_plotfigure(name='Velocity', figno=1)

    # Axes for velocity
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = xlimits
    # plotaxes.ylimits = [-0.5, 0.5]
    plotaxes.title = 'Velocity'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = velocity
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}

    # Figure for concentration
    plotfigure = plotdata.new_plotfigure(name='Concentration', figno=2)

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = xlimits
    plotaxes.title = 'Concentration'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = concentration
    plotitem.color = 'k'
    plotitem.kwargs = {'linewidth':3}

    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
