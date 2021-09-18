#
#
#
#

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
import scipy.interpolate as interpolate
import torch

np.set_printoptions(precision=4, linewidth=150, threshold=None, suppress=True)
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)
mpl.rcParams.update({'font.size': 20})
torch.set_printoptions(precision=4, linewidth=150, threshold=500000)


def make_visuals_grid(aml, data_source, noise_scale, device, number_of_points, maximal_number_of_iterations,
                      epoch, tb_writer):

    title = data_source.title
    visualization_type = data_source.visualization_type
    visualization_dim_inds = data_source.visualization_dim_inds
    visualization_axes_names = data_source.visualization_axes_names

    print('-.-.-.-.-.-.-.-.-.-.-.-.-.- make_visuals_grid', title)
    batch_size = number_of_points
    wide_scale = 4.0

    # Collect data.
    # Top row data -- from the source.
    on_data = data_source.get_batch(batch_size, device,
                                    on_manifold=True,
                                    noise_scale=noise_scale).detach()
    on_data_wide = data_source.get_batch(batch_size, device,
                                         on_manifold=True,
                                         noise_scale=noise_scale,
                                         initial_values_scale=wide_scale).detach()
    off_data = data_source.get_batch(batch_size, device,
                                     on_manifold=False,
                                     noise_scale=noise_scale).detach()

    # Subsequent rows - where all relations hold.
    combined_preimage_data = get_preimage_from_data_source(
        aml.new_g_nn, aml.gj_nns, data_source, device,
        number_of_points, maximal_number_of_iterations,
        initial_values_scale=1, noise_scale=noise_scale, away=False)
    combined_preimage_data_wide = get_preimage_from_data_source(
        aml.new_g_nn, aml.gj_nns, data_source, device,
        number_of_points, maximal_number_of_iterations,
        initial_values_scale=wide_scale, noise_scale=noise_scale, away=False)
    combined_nonpreimage_data = get_preimage_from_data_source(
        aml.new_g_nn, aml.gj_nns, data_source, device,
        number_of_points, maximal_number_of_iterations,
        initial_values_scale=1, noise_scale=noise_scale, away=True)

    # Where individual gs hold.
    g_preimage_data_all = []
    g_preimage_data_wide_all = []
    g_nonpreimage_data_all = []
    for j in range(len(aml.gj_nns) + 1):
        gj_nn = aml.gj_nns[j] if j < len(aml.gj_nns) else aml.new_g_nn
        if len(aml.gj_nns) > 0:
            g_preimage_data_all.append(get_preimage_from_data_source(
                gj_nn, [], data_source, device,
                number_of_points, maximal_number_of_iterations,
                initial_values_scale=1, noise_scale=noise_scale)
            )
            g_preimage_data_wide_all.append(get_preimage_from_data_source(
                gj_nn, [], data_source, device,
                number_of_points, maximal_number_of_iterations,
                initial_values_scale=wide_scale, noise_scale=noise_scale)
            )
            g_nonpreimage_data_all.append(get_preimage_from_data_source(
                gj_nn, [], data_source, device,
                number_of_points, maximal_number_of_iterations,
                initial_values_scale=1, noise_scale=noise_scale, away=True)
            )
        else:
            g_preimage_data_all.append(combined_preimage_data)
            g_preimage_data_wide_all.append(combined_preimage_data_wide)
            g_nonpreimage_data_all.append(combined_nonpreimage_data)

    # Now visualize.

    # Make figure
    figure_grid = gridspec.GridSpec(2 + aml.number_of_relations(), 3)
    figure = plt.figure(figsize=(15, 10 + (aml.number_of_relations()) * 5),
                        dpi=100)

    # Make the top row with the data from the data source.
    data_bounds = visualize_plots(figure, figure_grid[0, 0], on_data.cpu().numpy(),
                                  'on-manifold test data',
                                  visualization_type,
                                  visualization_dim_inds,
                                  visualization_axes_names)
    data_bounds_wide = visualize_plots(figure, figure_grid[0, 1],
                                       on_data_wide.cpu().numpy(),
                                       'on-manifold test data wider',
                                       visualization_type,
                                       visualization_dim_inds,
                                       visualization_axes_names,
                                       draw_color_bar=True)
    data_bounds_off = visualize_plots(figure, figure_grid[0, 2],
                                      off_data.cpu().numpy(),
                                      'off-manifold test data',
                                      visualization_type,
                                      visualization_dim_inds,
                                      visualization_axes_names)

    # Make caption title TeX.
    caption_title = r'$g_1'
    for j in range(len(aml.gj_nns)):
        caption_title += r' \cap g_' + str(j + 2)
    caption_title += r'$'

    visualize_plots(figure, figure_grid[1, 0], combined_preimage_data,
                    caption_title,
                    visualization_type,
                    visualization_dim_inds,
                    visualization_axes_names,
                    data_bounds=data_bounds)
    visualize_plots(figure, figure_grid[1, 1], combined_preimage_data_wide,
                    caption_title,
                    visualization_type,
                    visualization_dim_inds,
                    visualization_axes_names,
                    data_bounds=data_bounds_wide)
    visualize_plots(figure, figure_grid[1, 2], combined_nonpreimage_data,
                    'combo_nonpreim',
                    visualization_type,
                    visualization_dim_inds,
                    visualization_axes_names,
                    data_bounds=data_bounds_off)
    for j in range(len(aml.gj_nns) + 1):
        g_preim_data = g_preimage_data_all[j]
        g_preim_data_wide = g_preimage_data_wide_all[j]
        g_nonpreim_data = g_nonpreimage_data_all[j]
        visualize_plots(figure, figure_grid[2 + j, 0], g_preim_data,
                        r'$g_{:d}$'.format(j + 1),
                        visualization_type,
                        visualization_dim_inds,
                        visualization_axes_names,
                        data_bounds=data_bounds)
        visualize_plots(figure, figure_grid[2 + j, 1], g_preim_data_wide,
                        r'$g_{:d}$'.format(j + 1),
                        visualization_type,
                        visualization_dim_inds,
                        visualization_axes_names,
                        data_bounds=data_bounds_wide)
        visualize_plots(figure, figure_grid[2 + j, 2], g_nonpreim_data,
                        r'g{:d}_nonpreim'.format(j + 1),
                        visualization_type,
                        visualization_dim_inds,
                        visualization_axes_names,
                        data_bounds=data_bounds_off)
    # plt.savefig('/tmp/quiver_out.png', dpi=100)
    plt.tight_layout()

    # Output the figure.
    print('visualized', title)
    tb_writer.add_figure(title, figure, epoch)  # do it twice to make
    tb_writer.add_figure(title, figure, epoch)  # it show right away

    return figure


def visualize_plots(figure, figure_grid_cell, preimages,  title,
                    visualization_type,
                    visualization_dim_inds,
                    visualization_axes_names,
                    data_bounds=None,
                    color_bounds=(-1.0, 3.0), draw_color_bar=True):
    if visualization_type == '3D scatter':
        return visualize_scatter(figure, figure_grid_cell, preimages,
                                 title=title, data_bounds=data_bounds,
                                 visualization_dim_inds=visualization_dim_inds,
                                 visualization_axes_names=visualization_axes_names,
                                 dim=3)

    elif visualization_type == '2D scatter':
        return visualize_scatter(figure, figure_grid_cell, preimages,
                                 title=title, data_bounds=data_bounds,
                                 visualization_dim_inds=visualization_dim_inds,
                                 visualization_axes_names=visualization_axes_names,
                                 dim=2)
    elif visualization_type == '2D forward':
        return visualize_2D_forward(figure, figure_grid_cell, preimages,
                                    color_bounds=color_bounds,
                                    draw_color_bar=draw_color_bar,
                                    title=title, data_bounds=data_bounds,
                                    visualization_dim_inds=visualization_dim_inds,
                                    visualization_axes_names=visualization_axes_names)
    else:
        print(f'Visualisation type {visualization_type} is not yet implemented.')


def visualize_2D_forward(figure, figure_grid_cell, data,
                         visualization_axes_names=None, title=None,
                         data_bounds=None, color_bounds=(-1.0, 3.0),
                         draw_color_bar=True,
                         visualization_dim_inds=(0, 1, 2, 3)):

    min_number = 10

    x_0, y_0 = data[:, visualization_dim_inds[0]], data[:, visualization_dim_inds[1]]
    x_1,  y_1 = data[:, visualization_dim_inds[2]], data[:, visualization_dim_inds[3]]

    if data is None:
        print('visualize_vector_field '+title+': empty vector field.')
        return
    if data.shape[0] < min_number:
        print('visualize_vector_field '+title+': almost empty vector field.')
        return

    axes = figure.add_subplot(figure_grid_cell)

    dx, dy = x_1-x_0, y_1-y_0
    arrow_colors = dy/np.clip(dx, 1e-6, 1e6)
    if color_bounds is None:
        color_bounds = [arrow_colors.min(), arrow_colors.max()]

    color_map_normalization = mpl.colors.Normalize(vmin=color_bounds[0],
                                                   vmax=color_bounds[1],
                                                   clip=False)
    # was: scale=100.0 width=0.05  # scale=25.0 for 45Tn # scale=15.0 for Bullet
    image = axes.quiver(x_0, y_0, dx, dy, arrow_colors, units='xy', angles='xy',
                        scale=25.0, cmap=plt.cm.brg,
                        norm=color_map_normalization,
                        alpha=0.3, width=0.015, headlength=4)
    #  https://matplotlib.org/3.1.1/gallery/axes_grid1/
    #  demo_colorbar_with_inset_locator.html
    if draw_color_bar:
        figure.colorbar(image, ax=axes,
                        fraction=0.05, pad=0, shrink=0.5)
    plt.axis('equal')  # set this to see quiver without rescaling
    if x_0.shape[0] > min_number:
        try:
            # https://stackoverflow.com/questions/35877478/
            # matplotlib-using-1-d-arrays-in-streamplot
            xi = np.linspace(x_0.min(), x_0.max(), 100)
            yi = np.linspace(y_0.min(), y_0.max(), 100)
            X, Y = np.meshgrid(xi, yi)  # capitalized variables are 2D arrays
            DX = interpolate.griddata((x_0, y_0), dx, (X, Y), method='cubic')
            DY = interpolate.griddata((x_0, y_0), dy, (X, Y), method='cubic')
            plt.streamplot(X, Y, DX, DY, linewidth=1.5, color='k')
        except Exception:
            import traceback
            import logging
            logging.error(traceback.format_exc())

    if visualization_axes_names is not None:
        x_dimension_name, y_dimension_name = visualization_axes_names
        axes.set_xlabel(x_dimension_name)
        axes.set_ylabel(y_dimension_name)
    if title is not None:
        axes.set_title(title)
    if data_bounds is None:
        data_bounds = ((x_0.min(), x_0.max()), (y_0.min(), y_0.max()))

    axes.set_xlim(data_bounds[0])
    axes.set_ylim(data_bounds[1])

    return data_bounds  # [(x_0.min(), x_0.max()), (y_0.min(), y_0.max())]


def visualize_scatter(figure, figure_grid_cell, preimages,
                      data_bounds, title,
                      visualization_axes_names,
                      visualization_dim_inds=None,
                      dim=2):
    if visualization_dim_inds is None:
        if dim == 2:
            visualization_dim_inds = (0, 1)
        else:
            visualization_dim_inds = (0, 1, 2)

    if preimages is None:
        print('visualize_plots ' + title + ': empty preimage')
        return

    assert(len(visualization_dim_inds) >= dim)

    if dim == 2:
        axes = figure.add_subplot(figure_grid_cell)
        x_ind, y_ind = visualization_dim_inds[:2]
        axes.scatter(preimages[:, x_ind], preimages[:, y_ind],
                     s=1.0)
    else:
        axes = figure.add_subplot(figure_grid_cell, projection='3d')
        x_ind, y_ind, z_ind = visualization_dim_inds[:3]
        axes.scatter(preimages[:, x_ind], preimages[:, y_ind],
                     preimages[:, z_ind], s=1.0)

    if data_bounds is None:
        data_bounds = [None]*dim
        for i in range(dim):
            margin_ratio = 1.2
            data_bounds[i] = (preimages[:, visualization_dim_inds[i]].min() *
                              margin_ratio,
                              preimages[:, visualization_dim_inds[i]].max() *
                              margin_ratio)

    if visualization_axes_names is not None:
        x_dimension_name, y_dimension_name = visualization_axes_names[:2]
        axes.set_xlabel(x_dimension_name)
        axes.set_ylabel(y_dimension_name)
        if dim >= 3:
            z_dimension_name = visualization_axes_names[2]
            axes.set_zlabel(z_dimension_name)

    axes.set_xlim(data_bounds[0])
    axes.set_ylim(data_bounds[1])
    if dim >= 3:
        axes.set_zlim(data_bounds[2])

    if title is not None:
        axes.set_title(title)

    return data_bounds


def visualize_animation(title, figure, number_of_points, epoch,
                        save_path, tb_writer):
    print('visualize_animation', end='')

    def anim_angle(i):
        angle = i*20  # 10
        print(' {:d}'.format(angle), end='')
        sys.stdout.flush()
        for ax in figure.axes:
            ax.view_init(elev=20.0, azim=angle)
        return figure,

    def d_init():
        return figure,
    figure.set_dpi(30 if number_of_points < 5000 else 100)
    anim = animation.FuncAnimation(
        figure, anim_angle, init_func=d_init, frames=18, interval=1, blit=True)
    fnm = os.path.join(save_path, 'aml.gif')
    anim.save(fnm, fps=4, writer='imagemagick')
    print('.')
    from PIL import Image, ImageSequence
    img = Image.open(fnm)

    frames = np.array([np.array(frame.copy().convert('RGB').getdata(),
                                dtype=np.uint8).reshape((frame.size[1],
                                                         frame.size[0],
                                                         3))
                       for frame in ImageSequence.Iterator(img)])
    frames = np.expand_dims(frames.swapaxes(3, 1).swapaxes(2, 3), axis=0)
    print('add_video frames', frames.shape)
    tb_writer.add_video(title + '_gif', frames, epoch, fps=2)  # do it twice to
    tb_writer.add_video(title + '_gif', frames, epoch, fps=2)  # make it show right away


def pprint(msg, tensor):
    print(msg, end=' ')
    print(tensor.squeeze().detach().cpu().numpy())


def should_keep(output, on_output, off_output, away):
    """Finds indexes in output where values are close to/far from zero
    as measured by the ize of on_output/off_output.

    Returns a torch array."""
    if away:  # get indexes of data NOT in the preimage of 0
        margin = torch.abs(off_output).mean()
        keep_ids = torch.abs(output) > 0.5 * margin
    else:     # get indexes of data in the preimage of 0
        margin = torch.abs(on_output).mean()
        keep_ids = torch.abs(output) < 1.5 * margin
    return keep_ids.squeeze(-1)


def get_preimage_from_data(g_nn, gj_nns, on_inputs, off_inputs, viz_max_pts,
                           away=False):
    """"Returns numpy array of data inputs which lie in the (approximate)
     joint preimage of all the g_js  and the current g."""
    inp = off_inputs
    #  might need to supplement with inp = torch.cat([off_inputs, on_inputs], dim=0)
    keep_ids = None
    for j in range(len(gj_nns)+1):
        current_g_nn = gj_nns[j] if j < len(gj_nns) else g_nn
        on_out, off_out = current_g_nn(on_inputs), current_g_nn(off_inputs)
        current_keep_ids = should_keep(off_out, on_out, off_out, away).detach()
        keep_ids = current_keep_ids if keep_ids is None \
            else (keep_ids & current_keep_ids)

    if not keep_ids.any():
        return None  # empty preimage
    preimage_data = inp[keep_ids]
    if preimage_data.size(0) > viz_max_pts:
        perm = torch.randperm(preimage_data.size(0))
        preimage_data = preimage_data[perm[:viz_max_pts]]
    return preimage_data.detach().cpu().numpy()


def get_preimage_from_data_source(g_nn, gj_nns, data_source, device,
                                  number_of_points,
                                  maximal_number_of_iterations,
                                  initial_values_scale, noise_scale,
                                  away=False):
    """"Returns numpy array of data inputs which lie in the (approximate)
     joint preimage of all the g_js  and the current g."""
    batch_size = number_of_points

    on_data = data_source.get_batch(batch_size, device, on_manifold=True,
                                    noise_scale=noise_scale,
                                    initial_values_scale=initial_values_scale).detach()
    preimages = None
    for _ in range(maximal_number_of_iterations):
        current_off_data = data_source.get_batch(batch_size*10, device,
                                                 on_manifold=False,
                                                 noise_scale=noise_scale,
                                                 initial_values_scale=initial_values_scale).detach()
        current_preimages = get_preimage_from_data(g_nn, gj_nns,
                                                   on_data, current_off_data,
                                                   number_of_points, away)
        if preimages is None:
            preimages = current_preimages
        elif current_preimages is not None:
            preimages = np.vstack([preimages, current_preimages])

        if preimages is not None:
            print('Got {:d} preimages.'.format(preimages.shape[0], end=''))
            if preimages.shape[0] >= number_of_points:
                break
    print('')
    return preimages
