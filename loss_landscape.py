# Copied from https://github.com/tomgoldstein/loss-landscape with some minor modifications

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import neural_utils as nu

################################################################################
#                 Supporting functions for weights manipulation
################################################################################
def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w + torch.Tensor(d).type(type(w))


def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()) for w in weights]


def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]


################################################################################
#                        Normalization Functions
################################################################################
def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def ignore_biasbn(directions):
    """ Set bias and bn parameters in directions to zero """
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)

################################################################################
#                       Create directions
################################################################################
def create_target_direction(net, net2):
    """
        Setup a target direction from one model to the other

        Args:
          net: the source model
          net2: the target model with the same architecture as net.
          dir_type: 'weights' or 'states', type of directions.

        Returns:
          direction: the target direction from net to net2 with the same dimension
                     as weights or states.
    """

    assert (net2 is not None)
    # direction between net2 and net
    w = get_weights(net)
    w2 = get_weights(net2)
    direction = get_diff_weights(w, w2)


    return direction


def create_random_direction(net, ignore='biasbn', norm='filter'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    weights1 = get_weights(net) # a list of parameters.
    direction1 = get_random_weights(weights1)
    normalize_directions_for_weights(direction1, weights1, norm, ignore)

    weights2 = get_weights(net) # a list of parameters.
    direction2 = get_random_weights(weights2)
    normalize_directions_for_weights(direction2, weights2, norm, ignore)


    return [direction1, direction2]

################################################################################
#                       Compute Surface
################################################################################

def compute(
    net,
    weight,
    d,
    x_range,
    y_range,
    X_data,
    U_data,
    X_col,
    f,
    lambda_phys
):
    # grid args like (x_min, x_max, x_num, y_min, y_max, y_num)
    # without h5 file infrastructure
    # s is deepcopy of net state dict

    x_min, x_max, x_num = x_range
    y_min, y_max, y_num = y_range
    x = np.linspace(x_min, x_max, x_num)
    y = np.linspace(y_min, y_max, y_num)

    X,Y = np.meshgrid(x, y, indexing="ij")

    # evaluate losses

    coords = np.stack([X.flatten(), Y.flatten()], axis=1)

    losses = np.zeros(len(coords))
    for k, coord in enumerate(coords):
        set_weights(net, weight, d, coord)
        loss = nu.data_loss(net, X_data, U_data) + lambda_phys*nu.physics_loss(net, X_col, f)

        losses[k] = loss.item()

    losses = losses.reshape(X.shape)

    set_weights(net, weight)

    return X, Y, losses

def plot(X,Y,losses, show = True, vmax=1):
    
    fig_contour = plt.figure()
    contour = plt.contour(X, Y, losses, cmap='summer', vmax=vmax)
    plt.clabel(contour, inline=1, fontsize=8)

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    fig_heatmap = plt.figure()
    heatmap = plt.imshow(losses, cmap='viridis', origin='lower', vmax=vmax)
    plt.colorbar(heatmap)

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    fig_surface = plt.figure()
    ax = fig_surface.add_subplot(111, projection='3d')
    norm = plt.Normalize(vmin=losses.min(), vmax=vmax)
    surf = ax.plot_surface(X, Y, losses, cmap='plasma', linewidth=0, antialiased=False, norm=norm)
    fig_surface.colorbar(surf, shrink=0.5, aspect=15)

    if show: plt.show()

    return [fig_contour, fig_heatmap, fig_surface]