#
# 
#
# Simple data generators for illustrating relationship learning with AML.
#
# import time
import torch
import math
import numpy as np
from functools import partial
from scipy.integrate import solve_ivp
from numbers import Number, Real
from abc import ABC, abstractmethod
from numpy.random import default_rng

from aml.aml_visualize import make_visuals_grid, visualize_animation

#  Some utilities.
rng = default_rng()  # create a random number generator


def to_torch(data, device):
    return torch.from_numpy(data).float().to(device)


def row_product(vector, matrix):
    return (matrix.T*vector).T


def row_sum(vector, matrix):
    return (matrix.T+vector).T


class DataSource(ABC):
    def __init__(self, batch_size, device, dim_names=None,
                 visualization_dim_inds=(0, 1),
                 visualization_type='2D scatter', 
                 visualization_axes_names=None):
        self.device = device
        self.batch_size = batch_size
        self.title = self.__class__.__name__

        if dim_names is None:
            self.dim_names = self._dim_names()
        else:
            self.dim_names = dim_names

        if visualization_dim_inds is None:
            visualization_dim_inds = (0, 1)
        self.visualization_dim_inds = visualization_dim_inds
        if visualization_type is None:
            visualization_type = (0, 1)
        self.visualization_type = visualization_type
        if visualization_axes_names is None:
            print(f'vdi={visualization_dim_inds}, dn={self.dim_names}')
            self.visualization_axes_names = [self.dim_names[i] for i in 
                                             visualization_dim_inds]
        else:
            self.visualization_axes_names = visualization_axes_names

    @abstractmethod
    def _dim_names(self):
        pass

    @abstractmethod
    def get_batch(self, batch_size, device='cpu', on_manifold=True,
                  noise_scale=0.0):
        pass

    def visualize(self, aml,  noise_scale, number_of_points,
                  maximal_number_of_iterations,
                  epoch, save_path, tb_writer, animation=False):

        figure = make_visuals_grid(aml, self, noise_scale, self.device,  
                                   number_of_points, 
                                   maximal_number_of_iterations, 
                                   epoch, tb_writer)
        if animation:
            if self.visualization_type == '3D scatter':
                visualize_animation(self.title, figure, number_of_points, 
                                    epoch, save_path, tb_writer)
            else:
                print(f'Animation for {self.visualization_type}' +
                      f' is not currently supported.')


def process_data(data, on_manifold, noise_scale, device):
    mins, maxs = data.min(axis=0), data.max(axis=0)
    if not on_manifold:
        data = naive_off_manifold(data, mins, maxs)
    data = add_noise(data, mins, maxs, noise_scale)
    return to_torch(data, device)


def naive_off_manifold(data, mins, maxs):
    margin_padding = 1
    return (rng.random(size=data.shape)*(maxs-mins) + mins)*margin_padding


def add_noise(data, mins, maxs, noise_scale):
    data += (rng.random(size=data.shape)-0.5)*(maxs-mins)*noise_scale
    return data


class ParametricCurve(DataSource):
    def __init__(self, batch_size, device,
                 parametric_equations, domain_segment, dim_names=None,
                 defining_equations=None,
                 parametric_vectorized=False,
                 defining_tensor_vectorized=False,
                 visualization_dim_inds=(0, 1),
                 visualization_type='2D scatter'):
        super().__init__(batch_size, device, dim_names,
                         visualization_dim_inds,
                         visualization_type)
        self.equations = parametric_equations  # list of functions of t
        self.dim = len(self.equations)
        self.domain_start, self.domain_end = domain_segment
        self.domain_length = self.domain_end-self.domain_start
        if defining_equations is not None:
            self.defining_equations = defining_equations

        self.parametric_vectorized = parametric_vectorized
        self.defining_tensor_vectorized = defining_tensor_vectorized

    def _dim_names(self):
        return ['w'+str(i) for i in range(self.dim)]

    def get_batch(self, batch_size=None, device=None, on_manifold=True,
                  noise_scale=0.0, initial_values_scale=None):
        if device is None:
            device = self.device
        if batch_size is None:
            batch_size = self.batch_size
        # get random points in the domain
        ts = rng.random(size=batch_size)*self.domain_length+self.domain_start

        data = np.empty(shape=(batch_size, self.dim))
        if self.parametric_vectorized:
            for j, f in enumerate(self.equations):
                data[:, j] = f(ts)
        else:
            for i, t in enumerate(ts):
                datum = [f(t) for f in self.equations]
                data[i, :] = datum

        return process_data(data, on_manifold, noise_scale, device)

    def evaluate_gs(self, data, number_of_gs=None):
        if number_of_gs is None:
            number_of_gs = len(self.defining_equations)
        if number_of_gs > len(self.defining_equations):
            number_of_gs = len(self.defining_equations)

        g_values = torch.empty(size=(data.shape[0], number_of_gs))
        if self.defining_tensor_vectorized:
            for eq_number in range(number_of_gs):
                g_values[:, eq_number] = \
                    self.defining_equations[eq_number](data)

        else:
            for i in range(data.shape[0]):
                datum = data[i, :]
                for eq_number in range(number_of_gs):
                    g_values[i, eq_number] = \
                        self.defining_equations[eq_number](*datum)

        return g_values


class Ellipse(ParametricCurve):
    """Data from an ellipse curve in 3D

    A simple analytic example: on-manifold data lies on a curve
    given in parametric form by the equation
    w(t) = sqrt(2)*cos(t), 2*sin(t)-1, sin(t)-1
    for t in [0,2*pi].
    Full arc is the whole ellipse, but we can also do partial arcs.

    This is the intersection of a plane and a plane and a hyperboloid:
    y - 2z - 1 = 0
    x^2 + y^2 - 2*z^2 - 1 = 0 """
    def __init__(self, batch_size, device='cpu', arc_fraction=1,
                 parametric_vectorized=True,
                 defining_tensor_vectorized=True,
                 visualization_dim_inds=(0, 1, 2),
                 visualization_type='3D scatter'):
        if visualization_dim_inds is None:
            visualization_dim_inds = (0, 1, 2)
        if visualization_type is None:
            visualization_type = '3D scatter'

        domain_segment = [0,  2*np.pi*arc_fraction]

        if defining_tensor_vectorized:
            def plane(data):
                y, z = data[:, 1], data[:, 2]
                return y - 2*z - 1

            def hyperboloid(data):
                x, y, z = data[:, 0], data[:, 1], data[:, 2]
                return x**2 + y**2 - 2*z**2 - 1
        else:
            def plane(_, y, z):
                return y - 2*z - 1

            def hyperboloid(x, y, z):
                return x**2 + y**2 - 2*z**2 - 1
        defining_equations = [plane, hyperboloid]

        if parametric_vectorized:
            def x_ell(ts):
                return math.sqrt(2)*np.cos(ts)

            def y_ell(ts):
                return 2*np.sin(ts)-1

            def z_ell(ts):
                return np.sin(ts)-1
        else:
            def x_ell(t):
                return math.sqrt(2)*math.cos(t)

            def y_ell(t):
                return 2*math.sin(t)-1

            def z_ell(t):
                return math.sin(t)-1
        parametric_equations = [x_ell, y_ell, z_ell]

        super().__init__(batch_size, device,
                         parametric_equations, domain_segment,
                         dim_names=('x', 'y', 'z'),
                         defining_equations=defining_equations,
                         parametric_vectorized=parametric_vectorized,
                         defining_tensor_vectorized=defining_tensor_vectorized,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_type=visualization_type
                         )

    def _dim_names(self):
        assert False, 'Called _dim_names of Ellipse class.'


class ForwardEvolution(DataSource):
    """Use forward_map to generate data."""
    def __init__(self, batch_size, device, forward_map,
                 initial_value_ranges, parameter_ranges=None,
                 start_time=0, end_time=1, time_steps=None,
                 post_process=None, output_dimension=None,
                 vectorized=False, dim_names=None,
                 visualization_dim_inds=(0, 1, 2, 3),
                 visualization_type='2D forward',
                 visualization_axes_names=None):

        if visualization_dim_inds is None:
            visualization_dim_inds = (0, 1, 2, 3),
        if visualization_type is None:
            visualization_type = '2D forward'

        if visualization_axes_names is None:
            x_axis_name = self.dim_names(self.visualization_dim_inds[0]) +\
                          self.dim_names(self.visualization_dim_inds[2])
            y_axis_name = self.dim_names(self.visualization_dim_inds[1]) +\
                          self.dim_names(self.visualization_dim_inds[3])
            visualization_axes_names = [x_axis_name, y_axis_name]

        if isinstance(initial_value_ranges, Number):
            initial_value_ranges = (initial_value_ranges,)
        if isinstance(parameter_ranges, Number):
            parameter_ranges = (parameter_ranges,)

        self.forward_map = forward_map
        self.dim = len(initial_value_ranges)
        if output_dimension is not None:
            self.output_dimension = output_dimension
        else:
            self.output_dimension = self.dim

        self.initial_value_ranges = initial_value_ranges
        self.parameter_ranges = parameter_ranges
        self.post_process = post_process

        if time_steps is None:
            self.time_steps = (start_time, end_time)
        else:
            self.time_steps = time_steps
        self.time_steps = np.array(self.time_steps)

        self.start_time = self.time_steps[0]
        self.end_time = self.time_steps[-1]
        self.vectorized = vectorized

        self.visualization_dim_inds = visualization_dim_inds
        self.visualization_type = visualization_type

        super().__init__(batch_size, device, dim_names,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_type=visualization_type,
                         visualization_axes_names=visualization_axes_names)

    def _dim_names(self):
        return [f'w_{d}_t={step}'
                for d in range(self.output_dimension)
                for step in self.time_steps]

    def get_batch(self, batch_size=None, device=None, on_manifold=True,
                  noise_scale=0.0, initial_values_scale=1.0):

        if device is None:
            device = self.device
        if batch_size is None:
            batch_size = self.batch_size

        # Helper data and parameter acquisition functions.
        def get_parameters(parameters_shape, b_size=1):
            params = np.empty(shape=parameters_shape)
            for j, parameter_range in enumerate(self.parameter_ranges):
                try:
                    a, b = parameter_range
                    params[..., j] = rng.random(b_size)*(b-a)+a
                except TypeError:
                    params[..., j] = parameter_range*np.ones(b_size)
            return params

        def get_data(initial_values_shape, parameters_shape, b_size=1):
            # Make parameters.
            parameters = None
            if self.parameter_ranges:
                parameters = get_parameters(parameters_shape,
                                            b_size)
            # Make initial values.
            initial_values = np.empty(shape=initial_values_shape)
            for d, initial_value_range in enumerate(self.initial_value_ranges):
                try:
                    a, b = initial_value_range
                    initial_values[..., d] = (rng.random(b_size)*(b-a)+a) *\
                                             initial_values_scale
                except TypeError:
                    initial_values[..., d] = np.ones(b_size) * \
                                             initial_value_range * \
                                             initial_values_scale

            # Generate data.
            dat = self.forward_map(self.time_steps, initial_values, parameters)
            if self.post_process is not None:
                dat = self.post_process(dat, parameters)
            return dat

        # Vectorized flag controls data acquisition mode.
        if self.vectorized:
            data = get_data((batch_size, self.dim),
                            (batch_size, len(self.parameter_ranges)),
                            batch_size)
            data = data.reshape(batch_size, -1)

        else:
            data = np.empty(shape=(batch_size,
                                   self.output_dimension*len(self.time_steps)))
            for i in range(batch_size):
                data_i = get_data(self.dim, len(self.parameter_ranges))
                data_i = data_i.flatten()
                data[i, :] = data_i

        return process_data(data, on_manifold, noise_scale, device)


# Forward maps for different dynamical systems.
def flow_forward_map(time_steps, initial_values,  parameters, flow_function):
    ys = np.empty(shape=(len(initial_values), len(time_steps)))
    for time_step, t in enumerate(time_steps):
        y = flow_function(t, initial_values, *parameters)
        ys[:, time_step] = y
    return ys


def ode_forward_map(time_steps, initial_values, parameters, ode_system):
    sol_i = solve_ivp(ode_system, (time_steps[0], time_steps[-1]),
                      initial_values, t_eval=time_steps, args=parameters)
    assert sol_i.success, sol_i.message
    return sol_i.y


def function_forward_map(initial_values, _, parameters, function):
    ys = np.empty(shape=(len(initial_values), 2))
    ys[:, 0] = initial_values
    ys[:, 1] = function(initial_values, *parameters)
    return ys


# Classes for different dynamical systems.

class ODEsystem(ForwardEvolution):
    def __init__(self, batch_size, device, ode_system,
                 initial_value_ranges, parameter_ranges=None,
                 start_time=0, end_time=1, time_steps=None,
                 post_process=None, output_dimension=None,
                 visualization_dim_inds=None,
                 visualization_type=None, visualization_axes_names=None):
        super().__init__(batch_size, device,
                         partial(ode_forward_map, ode_system=ode_system),
                         initial_value_ranges, parameter_ranges,
                         start_time, end_time, time_steps,
                         post_process, output_dimension,
                         visualization_dim_inds,
                         visualization_type,
                         visualization_axes_names)


class Flow(ForwardEvolution):
    def __init__(self, batch_size, device, flow_function,
                 initial_value_ranges, parameter_ranges=None,
                 start_time=0, end_time=1, time_steps=None,
                 post_process=None, output_dimension=None,
                 t_vectorized=False, vectorized=False,
                 visualization_dim_inds=None,
                 visualization_type=None,
                 visualization_axes_names=None):

        if t_vectorized or vectorized:
            forward_map = flow_function
        else:
            forward_map = partial(flow_forward_map, flow_function=flow_function)
        super().__init__(batch_size, device, forward_map,
                         initial_value_ranges, parameter_ranges,
                         start_time, end_time, time_steps,
                         post_process, output_dimension, vectorized,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_type=visualization_type,
                         visualization_axes_names=visualization_axes_names)


class Function(ForwardEvolution):
    def __init__(self, batch_size, device, function,
                 initial_value_ranges, parameter_ranges=None,
                 post_process=None, output_dimension=None,
                 visualization_dim_inds=None,
                 visualization_type=None,
                 visualization_axes_names=None):
        super().__init__(batch_size, device,
                         partial(function_forward_map, function=function),
                         initial_value_ranges, parameter_ranges,
                         post_process, output_dimension,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_type=visualization_type,
                         visualization_axes_names=visualization_axes_names,
                         )


# Block on incline classes.
class BlockOnInclineODE(ODEsystem):
    def __init__(self, batch_size, device='cpu', drag_type='None',
                 initial_value_ranges=((0, 1), (-1, 1)),
                 parameter_ranges=(9.8, 0.4, 0.2),
                 start_time=0, end_time=1, time_steps=None,
                 post_process=None, output_dimension=None,
                 visualization_dim_inds=None,
                 visualization_type=None, visualization_axes_names=None):

        if drag_type == 'Proportional':
            ode_system = block_on_incline_linear_drag_ODE
        elif drag_type == 'Quadratic':
            ode_system = block_on_incline_quadratic_drag_ODE
        else:
            ode_system = block_on_incline_no_drag_ODE

        super().__init__(batch_size, device, ode_system,
                         initial_value_ranges, parameter_ranges,
                         start_time, end_time, time_steps,
                         post_process, output_dimension,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_type=visualization_type,
                         visualization_axes_names=visualization_axes_names)


class BlockOnIncline(Flow):
    def __init__(self, batch_size, device='cpu', drag_type='None',
                 initial_value_ranges=((0, 1), (-1, 1)),
                 parameter_ranges=(9.8, 0.4, 0.2),
                 start_time=0, end_time=1, time_steps=None,
                 post_process=None, output_dimension=None, t_vectorized=False,
                 vectorized=True,
                 visualization_type=None,
                 visualization_axes_names=None,
                 visualization_dim_inds=None):

        if drag_type == 'Proportional':
            if vectorized:
                flow_function = block_on_incline_linear_drag_flow_vectorized
            elif t_vectorized:
                flow_function = block_on_incline_linear_drag_flow_t_vectorized
            else:
                flow_function = block_on_incline_linear_drag_flow
        elif drag_type == 'Quadratic':
            if vectorized:
                flow_function = block_on_incline_quadratic_drag_flow_vectorized
            elif t_vectorized:
                flow_function = block_on_incline_quadratic_drag_flow_t_vectorized
            else:
                flow_function = block_on_incline_quadratic_drag_flow
        else:
            if vectorized:
                flow_function = block_on_incline_no_drag_flow_vectorized
            elif t_vectorized:
                flow_function = block_on_incline_no_drag_flow_t_vectorized
            else:
                flow_function = block_on_incline_no_drag_flow

        if post_process is None and visualization_type is None:
            visualization_axes_names = ['position', 'velocity']
            visualization_dim_inds = [0, 1, 2, 3]
        elif post_process == ambient and visualization_type is None:
            visualization_axes_names = ['x', 'v_x']
            visualization_dim_inds = [0, 2, 4, 6]

        super().__init__(batch_size, device, flow_function,
                         initial_value_ranges, parameter_ranges,
                         start_time, end_time, time_steps,
                         post_process, output_dimension,
                         t_vectorized, vectorized,
                         visualization_type=visualization_type,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_axes_names=visualization_axes_names)

    def _dim_names(self):
        if self.post_process is None:
            return ['position_0', 'velocity_0', 'position_1',  'velocity_1']
        elif self.post_process == ambient:
            return ['x_position_0', 'y_position_0',
                    'x_velocity_0', 'y_velocity_0',
                    'x_position_1', 'y_position_1',
                    'x_velocity_1', 'y_velocity_1'
                    ]

        else:
            return super()._dim_names()


# Specialized block on incline classes.
class ConservativeBlockOnIncline(BlockOnIncline):
    def __init__(self, batch_size, device='cpu',
                 initial_value_ranges=((0, 1), (-1, 1)),
                 parameter_ranges=(9.8, (0.1, 1), 0),
                 start_time=0, end_time=1, time_steps=None,
                 post_process=None, output_dimension=None,
                 t_vectorized=False, vectorized=True,
                 visualization_type=None,
                 visualization_axes_names=None,
                 visualization_dim_inds=None
                 ):

        if isinstance(parameter_ranges, Number):
            self.g, self.theta = 9.8, parameter_ranges
        elif len(parameter_ranges) == 1:
            self.g, self.theta = 9.8, parameter_ranges[0]
        elif len(parameter_ranges) == 2:
            self.g, self.theta = parameter_ranges
        else:
            self.g, self.theta = parameter_ranges[0], parameter_ranges[1]

        if isinstance(self.theta, Real):
            self.sintheta = math.sin(self.theta)
            self.costheta = math.cos(self.theta)

        parameter_ranges = (self.g, self.theta, 0, 0)
        super().__init__(batch_size, device, 'None',
                         initial_value_ranges, parameter_ranges,
                         start_time, end_time, time_steps,
                         post_process, output_dimension,
                         t_vectorized, vectorized,
                         visualization_type=visualization_type,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_axes_names=visualization_axes_names)

    def evaluate_gs(self, data):
        if len(self.time_steps) != 2:
            print('More than two time steps.')
            return
        if not isinstance(self.g, Real):
            print('Variable gravity.')
            return
        if not isinstance(self.theta, Real):
            print('Variable angle.')

        if self.post_process is None:
            g_values = energy_difference_block(data, self.g, self.sintheta)
        elif self.post_process == ambient:
            g_values = energy_difference_block_ambient(data, self.g)
        else:
            print('This post processing is not impelmented.')
            g_values = None

        return g_values


class ConservativeBlock1D(ConservativeBlockOnIncline):
    def __init__(self, batch_size, device='cpu',
                 visualization_type=None,
                 visualization_axes_names=None,
                 visualization_dim_inds=None):
        super().__init__(batch_size, device,
                         visualization_type=visualization_type,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_axes_names=visualization_axes_names)


class ConservativeBlock2D(ConservativeBlockOnIncline):
    def __init__(self, batch_size, device='cpu',
                 visualization_type=None,
                 visualization_axes_names=None,
                 visualization_dim_inds=None):
        super().__init__(batch_size, device,
                         post_process=ambient, output_dimension=4,
                         visualization_type=visualization_type,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_axes_names=visualization_axes_names)


class ConservativeBlockConstVel1D(ConservativeBlockOnIncline):
    def __init__(self, batch_size, device='cpu',
                 visualization_type=None,
                 visualization_axes_names=None,
                 visualization_dim_inds=None):
        super().__init__(batch_size, device,
                         parameter_ranges=(0,),
                         visualization_type=visualization_type,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_axes_names=visualization_axes_names)


class ConservativeBlockConstVel2D(ConservativeBlockOnIncline):
    def __init__(self, batch_size, device='cpu',
                 visualization_type=None,
                 visualization_axes_names=None,
                 visualization_dim_inds=None):
        super().__init__(batch_size, device,
                         parameter_ranges=(0,),
                         post_process=ambient, output_dimension=4,
                         visualization_type=visualization_type,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_axes_names=visualization_axes_names)


class ConservativeBlock45Incline1D(ConservativeBlockOnIncline):
    def __init__(self, batch_size, device='cpu',
                 visualization_type=None,
                 visualization_axes_names=None,
                 visualization_dim_inds=None):
        super().__init__(batch_size, device, parameter_ranges=(np.pi/4,),
                         visualization_type=visualization_type,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_axes_names=visualization_axes_names)


class Block36InclineNoDrag1D(BlockOnIncline):
    def __init__(self, batch_size, device='cpu',
                 visualization_type=None,
                 visualization_axes_names=None,
                 visualization_dim_inds=None):
        super().__init__(batch_size, device,
                         parameter_ranges=(9.8, np.pi/5, 0.2),
                         visualization_type=visualization_type,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_axes_names=visualization_axes_names)


class Block1D(BlockOnIncline):
    def __init__(self, batch_size, device='cpu',
                 visualization_type=None,
                 visualization_axes_names=None,
                 visualization_dim_inds=None):
        super().__init__(batch_size, device,
                         parameter_ranges=(9.8, np.pi/5, 0.2, 1),
                         visualization_type=visualization_type,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_axes_names=visualization_axes_names)


class Block2D(BlockOnIncline):
    def __init__(self, batch_size, device='cpu',
                 visualization_type=None,
                 visualization_axes_names=None,
                 visualization_dim_inds=None):
        super().__init__(batch_size, device,
                         parameter_ranges=(9.8, np.pi/5, 0.2, 1),
                         post_process=ambient, output_dimension=4,
                         visualization_type=visualization_type,
                         visualization_dim_inds=visualization_dim_inds,
                         visualization_axes_names=visualization_axes_names)


# Post processing for ambient coordinates.
def ambient(ys, parameters):
    s, v = ys[0, :], ys[1, :]
    _, theta, _, _ = parameters
    cos, sin = math.cos(theta), math.sin(theta)
    return np.vstack((s*cos, -s*sin, v*cos, -v*sin))


def ambient_vectorized(ys, parameters):  # ys(batch, dim, times), p (batch, dim)
    s, v = ys[:, 0, :], ys[:, 1, :]
    thetas = parameters[:, 1]
    cos, sin = np.cos(thetas), np.sin(thetas)
    return np.stack(((cos*s.T).T, -(sin*s.T).T, (cos*v.T).T, -(sin*v.T).T),
                    axis=1)  # (batch, out_dim, times)


# Block energy
def energy_difference_block(data, g, sintheta):
    s_0, s_1 = data[:, 0], data[:, 1]
    v_0, v_1 = data[:, 2], data[:, 3]
    return g*sintheta*(s_1-s_0)+0.5*(v_0**2-v_1**2)


def energy_difference_block_ambient(data, g):
    y_0, y_1 = data[:, 2], data[:, 3]
    dx_0,  dx_1 = data[:, 4], data[:, 5]
    dy_0, dy_1 = data[:, 6], data[:, 7]
    return g*(y_0-y_1)+0.5*(dx_0**2+dy_0**2-dx_1**2-dy_1**2)


# Block flows. See the supplementary "block_on_incline" files for derivations.
def block_on_incline_no_drag_flow(t, initial_values, g, theta, mu, _=None):
    s, v = initial_values
    c = g*(math.sin(theta)-mu*math.cos(theta))
    return s+v*t+0.5*c*t, v+c*t


def block_on_incline_no_drag_flow_t_vectorized(ts, initial_values, parameters):
    g, theta, mu, _ = parameters
    s, v = initial_values
    c = g*(math.sin(theta)-mu*math.cos(theta))
    return np.vstack((s+v*ts+0.5*c*np.square(ts), v+c*ts))  # shape (n,t)


def block_on_incline_no_drag_flow_vectorized(ts, initial_values, parameters):
    gs, thetas, mus = parameters[:, 0], parameters[:, 1], parameters[:, 2]
    ss, vs = initial_values[:, 0], initial_values[:, 1]
    cs = gs*(np.sin(thetas)-mus*np.cos(thetas))
    ones = np.ones_like(ts)
    return np.stack((np.outer(ss, ones)+np.outer(vs, ts)
                     + 0.5*np.outer(cs, np.square(ts)),
                     np.outer(vs, ones)+np.outer(cs, ts)),
                    axis=1)  # shape=(batch_size, dim=2, t)


def block_on_incline_linear_drag_flow(t, initial_values, g, theta, mu, r):
    s, v = initial_values
    c = g*(math.sin(theta)-mu*math.cos(theta))
    v_terminal = c/r
    k = v_terminal - v
    exp = math.exp(-r*t)
    return s+v_terminal*t+k*(exp-1)/r, v_terminal-k*exp


def block_on_incline_linear_drag_flow_t_vectorized(ts, initial_values,
                                                   parameters):
    ts = np.array(ts)
    g, theta, mu, r = parameters
    s, v = initial_values
    c = g*(math.sin(theta) - mu*math.cos(theta))
    v_terminal = c/r
    k = v_terminal - v
    exp = np.exp(-r*ts)
    return np.vstack((s+v_terminal*ts+k*(exp-1)/r, v_terminal-k*exp))


def block_on_incline_linear_drag_flow_vectorized(ts, initial_values,
                                                 parameters):
    gs, thetas, mus,  = parameters[:, 0], parameters[:, 1], parameters[:, 2]
    rs = parameters[:, 3]
    ss, vs = initial_values[:, 0], initial_values[:, 1]
    cs = gs*(np.sin(thetas)-mus*np.cos(thetas))
    ones = np.ones_like(ts)
    vs_terminal = cs/rs
    ks = vs_terminal - vs
    exp = np.exp(-np.outer(rs, ts))
    return np.stack((np.outer(ss, ones)+np.outer(vs_terminal, ts) +
                     row_product((ks/rs), exp-1) +
                     np.outer(vs_terminal, ones) -
                    row_product(ks, exp)),
                    axis=1)  # shape (batch_size, dim=2, t)


def block_on_incline_quadratic_drag_flow(t, initial_values, g, theta, mu, r):
    s, v = initial_values
    c = g*(math.sin(theta)-mu*math.cos(theta))
    v_terminal = (c/r)**0.5
    scaled_t = r*v_terminal*t
    if v >= 0:
        tanh = math.tanh(scaled_t)
        sinh, cosh = math.sinh(scaled_t), math.cosh(scaled_t)
        v_t = v_terminal*(v + v_terminal*tanh)/(v_terminal + v*tanh)
        s_t = s + math.log(cosh+(v/v_terminal)*sinh)/r
    else:
        atan = math.atan(v/v_terminal)
        T = atan/(r*v_terminal)
        tan = math.tan(scaled_t)
        sin, cos = math.sin(scaled_t), math.cos(scaled_t)

        if t < -T:
            v_t = v_terminal*(v + v_terminal*tan)/(v_terminal - v*tan)
            s_t = s - math.log(cos - (v/v_terminal)*sin)/r
        else:
            combined = scaled_t+atan
            v_t = v_terminal*math.tanh(combined)
            s_t = s + (math.log(math.cosh(combined))
                       - 0.5*math.log(1+(v/v_terminal)**2))/r
    return s_t, v_t


def block_on_incline_quadratic_drag_flow_t_vectorized(ts, initial_values,
                                                      parameters):
    ts = np.array(ts)
    g, theta, mu, r = parameters
    s, v = initial_values
    c = g*(math.sin(theta)-mu*math.cos(theta))
    v_terminal = (c/r)**0.5
    scaled_ts = r*v_terminal*ts
    if v >= 0:
        tanh = np.tanh(scaled_ts)
        sinh, cosh = np.sinh(scaled_ts), np.cosh(scaled_ts)
        v_t = v_terminal*(v + v_terminal*tanh)/(v_terminal + v*tanh)
        s_t = s + np.log(cosh+(v/v_terminal)*sinh)/r
    else:
        atan = np.arctan(v/v_terminal)
        T = atan/(r*v_terminal)
        tan, sin, cos = np.tan(scaled_ts), np.sin(scaled_ts), np.cos(scaled_ts)

        v_t_before_threshold = v_terminal*(v+v_terminal*tan)/(v_terminal-v*tan)
        s_t_before_threshold = s - np.log(cos-(v/v_terminal)*sin)/r

        combined = scaled_ts+atan
        v_t_after_threshold = v_terminal*np.tanh(combined)
        s_t_after_threshold = s + (np.log(np.cosh(combined))
                                   - 0.5*np.log(1+(v/v_terminal)**2))/r

        s_t = np.where(ts < -T, s_t_before_threshold, s_t_after_threshold)
        v_t = np.where(ts < -T, v_t_before_threshold, v_t_after_threshold)

    return np.vstack((s_t, v_t))


def block_on_incline_quadratic_drag_flow_vectorized(ts, initial_values,
                                                    parameters):
    gs, thetas, mus,  = parameters[:, 0], parameters[:, 1], parameters[:, 2]
    rs = parameters[:, 3]

    ss, vs = initial_values[:, 0], initial_values[:, 1]
    cs = gs*(np.sin(thetas)-mus*np.cos(thetas))
    vs_terminal = (cs/rs)**0.5

    scaled_ts = np.outer(rs*vs_terminal, ts)  # shape=(batch, t)

    # Above terminal velocity:
    scaled_Ts = np.where(vs > vs_terminal,
                         np.arctanh(vs_terminal/vs), np.ones_like(vs))

    vts_above_terminal = row_product(vs_terminal,
                                     1/np.tanh(row_sum(scaled_Ts, scaled_ts)))

    dsts_above_terminal = row_product(1/rs,
                                      row_sum(- np.log(np.sinh(scaled_Ts)),
                                              np.log(np.sinh(row_sum(scaled_Ts,
                                                                     scaled_ts)))))

    # Below terminal velocity:
    scaled_Ts = np.where((0 <= vs) & (vs < vs_terminal),
                         np.arctanh(vs/vs_terminal), np.zeros_like(vs))
    vts_below_terminal = row_product(vs_terminal,
                                     np.tanh(row_sum(scaled_Ts, scaled_ts)))

    dsts_below_terminal = row_product(1/rs,
                                      row_sum(- np.log(np.cosh(scaled_Ts)),
                                              np.log(np.cosh(row_sum(scaled_Ts,
                                                                     scaled_ts)))))

    vts_positive = np.where(vs > vs_terminal,
                            vts_above_terminal.T, vts_below_terminal.T).T
    dsts_positive = np.where(vs > vs_terminal,
                             dsts_above_terminal.T, dsts_below_terminal.T).T

    # Negative v, small t.
    scaled_Ts = np.where(vs < 0, np.arctan(vs/vs_terminal), np.zeros_like(vs))

    vts_negative_small_t = row_product(vs_terminal,
                                       np.tan(row_sum(scaled_Ts, scaled_ts)))
    vts_negative_big_t = row_product(vs_terminal,
                                     np.tanh(row_sum(scaled_Ts, scaled_ts)))

    dsts_negative_small_t = -row_product(1/rs,
                                         row_sum(- np.log(np.cos(scaled_Ts)),
                                                 np.log(np.cos(row_sum(scaled_Ts,
                                                                       scaled_ts)))))
    dsts_negative_big_t = row_product(1/rs,
                                      row_sum(-0.5*np.log(1+(vs/vs_terminal)**2),
                                              np.log(np.cosh(row_sum(scaled_Ts,
                                                                     scaled_ts)))))

    vt_s_negative = np.where(scaled_ts < -np.outer(scaled_Ts, np.ones_like(ts)),
                             vts_negative_small_t, vts_negative_big_t)
    dst_s_negative = np.where(scaled_ts < -np.outer(scaled_Ts, np.ones_like(ts)),
                              dsts_negative_small_t, dsts_negative_big_t)

    vt_s = np.where(vs > 0, vts_positive.T, vt_s_negative.T).T
    dst_s = np.where(vs > 0, dsts_positive.T, dst_s_negative.T).T
    st_s = row_sum(ss, dst_s)

    return np.stack((st_s, vt_s), axis=1)


# Block ODEs.
def block_on_incline_no_drag_ODE(t, initial_values, g, theta, mu, _=None):
    s, v = initial_values
    c = g*(math.sin(theta)-mu*math.cos(theta))
    return v, c


def block_on_incline_linear_drag_ODE(t, initial_values, g, theta, mu, r):
    s, v = initial_values
    c = g*(math.sin(theta)-mu*math.cos(theta))
    return v, c-r*v


def block_on_incline_quadratic_drag_ODE(t, initial_values, g, theta, mu, r):
    s, v = initial_values
    c = g*(math.sin(theta)-mu*math.cos(theta))
    return v, c-r*v*abs(v)


# Simple functions for testing.
def quadratic(y, a, b, c):
    return a*y*y+b*y+c


def decay_ode(_, y):
    return -0.5 * y


def accelerate_ode(_, y, a):
    s, v = y
    return v, a


def accelerate_flow(t, y, a):
    s, v = y
    return s+v*t+a*t*t*0.5, v+a*t


def main():
    print('Ellipse vectorized')
    data_source = Ellipse(7)
    batch = data_source.get_batch()
    print(batch)
    g_levels = data_source.evaluate_gs(batch)
    print(f'g levels={g_levels}')

    print('Ellipse non-vectorized')
    data_source = Ellipse(7,
                          parametric_vectorized=False,
                          defining_tensor_vectorized=False)
    batch = data_source.get_batch()
    print(batch)
    g_levels = data_source.evaluate_gs(batch)
    print(f'g levels={g_levels}')

    data_source = ConservativeBlockConstVel1D(5)
    batch = data_source.get_batch()
    print(batch)
    en_diff = data_source.evaluate_gs(batch)
    print(f'energy differences 1={en_diff}')

    print('Quadratic non-vectorized')
    data_source = BlockOnIncline(1,
                                 initial_value_ranges=(1, -10),
                                 parameter_ranges=(9.8, 0.2, 0.2, 1),
                                 time_steps=(0, 1, 3, 5, 10),
                                 drag_type='Quadratic', vectorized=False,
                                 post_process=ambient, output_dimension=4)
    batch = data_source.get_batch()
    print(batch)
    print('Quadratic vectorized')
    data_source = BlockOnIncline(1,
                                 initial_value_ranges=(1, -10),
                                 parameter_ranges=(9.8, 0.2, 0.2, 1),
                                 time_steps=(0, 1, 3, 5, 10),
                                 drag_type='Quadratic', vectorized=True,
                                 post_process=ambient_vectorized,
                                 output_dimension=4)
    batch = data_source.get_batch()
    print(batch)


if __name__ == "__main__":
    main()
