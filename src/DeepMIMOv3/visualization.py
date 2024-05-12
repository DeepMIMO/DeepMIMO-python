import matplotlib.pyplot as plt
import numpy as np


def plot_LoS_status(bs_location, user_locations, user_LoS, scat_size='auto'):
    """
    Scatters the users and one basestations and colors the users based on their
    line-of-sight status.

    Parameters
    ----------
    bs_location : numpy array
        One dimensional array with the xy position of the basestation.
    user_locations : numpy array
        A matrix containing user locations across rows and xy positions across
        columns. Expected shape: <n_users> by 2
    user_LoS : numpy array
        One dimensional array with the LoS status of each user. The length
        should match the number of users in user_locations.
    scat_size : float, optional
        Size of the scatter points. The default is 'auto'.

    Returns
    -------
    None.

    """
    LoS_map = {-1: ('r', 'No Path'), 0: ('b', 'NLoS'), 1: ('g', 'LoS')}
    
    # Calculate scatter size based on point density
    if scat_size == 'auto':
        n_points = user_locations.shape[0]
        area = np.prod(np.max(user_locations, axis=0)[:2] - 
                       np.min(user_locations, axis=0)[:2])
        point_density = n_points / area
        scat_size = 1 / (100 * point_density)
    
    for unique_LoS_status in LoS_map.keys():
    # Plot different status one by one to assign legend labels
        users_w_key = user_LoS==unique_LoS_status
        plt.scatter(user_locations[users_w_key, 0], 
                    user_locations[users_w_key, 1], 
                    c=LoS_map[unique_LoS_status][0], 
                    label=LoS_map[unique_LoS_status][1], s=scat_size)
    plt.scatter(bs_location[0], bs_location[1], 
                c='k', marker='x', label='Basestation')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    lgd = plt.legend(framealpha=.9, loc='lower left')
    lgd.legend_handles[0]._sizes = [20]
    lgd.legend_handles[1]._sizes = [20]
    lgd.legend_handles[2]._sizes = [20]
    plt.xlim([user_locations[:, 0].min(), user_locations[:, 0].max()])
    plt.ylim([user_locations[:, 1].min(), user_locations[:, 1].max()])


def plot_coverage(rxs, cov_map, dpi=300, figsize=(6,4), cbar_title=None, title=False,
                  scat_sz=.5, bs_pos=None, bs_ori=None, legend=False, lims=None,
                  proj_3D=False, equal_aspect=False, tight=True, cmap='viridis'):
    """
    This function scatters the users' positions <rxs> and colors them with <cov_map>.

    Parameters
    ----------
    rxs : numpy.ndarray
        User position array. Dimensions: [n_users, 3].
    cov_map : tuple, list or numpy.ndarray
        Coverage map. Or a map of the feature based on which to color the user positions.
        Dimension: n_users
    dpi : int, optional
        Resolution. The default is 300.
    figsize : tuple, optional
        Figure (horizontal size, vertical size) in inches. The default is (6,4).
    cbar_title : string, optional
        Title/text of the colorbar. The default is None.
    title : string, optional
        Title of the plot. No title if None, empty string or False. The default is False.
    scat_sz : float, optional
        Scatter marker size. The default is .5.
    bs_pos : tuple, list or numpy.ndarray, optional
        Transmitter (considered the Base station) position. If valid (not None),
        it puts a 'x' marker in that [x,y (,z)] position. The default is None.
    bs_ori : tuple, list or numpy.ndarray, optional
        Transmitter (considered the Base station) orientation. If valid (not None),
        it draws a line with this direction, starting at the BS position. 
        Orientation/Rotation is around [x,y,z] following the right hand rule. 
        [0,0,0] = Antenna pointing towards +x
        [0,0,90] = Antenna pointing towards +y
        [0,90,0] = Antenna pointing towards -z
        Another way of interpreting: 
            z-rotation is azimuth (where 0 is +x)
            y-rotation is elevation/tilt (where 0 is the horizon)
        The default is None.
    legend : bool, optional
        Whether to include a plot legend. The default is False.
    lims : tuple, list or numpy.ndarray, optional
        Coverage color limits. Helps setting the limits of the colormap and colorbar. 
        The default is None.
    proj_3D : bool, optional
        Whether to make a 3D or 2D plot. True is 3D, False is 2D. The default is False.
    equal_aspect : bool, optional
        Whether to have axis with the same scale. Note: if not done with 
        precaution, it can ruin a 3D visualization. The default is False.
    tight : bool, optional
        Whether to set the plot xy(z) limits automatically based on the values
        so that the axis are maximally used and no data is omitted.
        The default is True.
    cmap : string, optional
        Colormap string identifier for matplotlib. 
        See available colormaps in: 
        https://matplotlib.org/stable/users/explain/colors/colormaps.html
        or by runnign: plt.colormaps.get_cmap('not a colormap')
        The default is 'viridis'.

    Returns
    -------
    fig : matplotlib figure
        fig as in "fig, ax = plt.subplots()"
        The motivation behind returning these elements is allowing editing and
        saving, before displaying (plt.show() is not used).
    ax : matplotlib axes
        ax as in "fig, ax = plt.subplots()"
    cbar : matplotlib colorbar
        The colorbar associated with the coverage map. 

    """
    
    plt_params = {'cmap': cmap}
    if lims:
        plt_params['vmin'], plt_params['vmax'] = lims[0], lims[1]
    
    n = 3 if proj_3D else 2 # n = coordinates to consider
    
    xyz = {s: rxs[:,i] for s,i in zip(['x', 'y', 'zs'], range(n))}
    
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize,
                           subplot_kw={'projection': '3d'} if proj_3D else {})
    
    im = plt.scatter(**xyz, c=cov_map, s=scat_sz, marker='s', **plt_params)

    cbar = plt.colorbar(im, label='Received Power [dBm]' if not cbar_title else cbar_title)
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # TX position
    if bs_pos is not None:
        ax.scatter(*bs_pos[:n], marker='P', c='r', label='TX')
    
    # TX orientation
    if bs_ori is not None and bs_pos is not None:
        r = 30 # ref size of pointing direction
        tx_lookat = np.copy(bs_pos)
        tx_lookat[:2] += r * np.array([np.cos(bs_ori[2]), np.sin(bs_ori[2])]) # azimuth
        tx_lookat[2] -= r / 10 * np.sin(bs_ori[1]) # elevation
        
        line_components = [[bs_pos[i], tx_lookat[i]] for i in range(n)]
        ax.plot(*line_components, c='k', alpha=.5, zorder=3)
        
    if title:
        ax.set_title(title)
    
    if legend:
        plt.legend(loc='upper center', ncols=10, framealpha=.5)
    
    if tight:
        s = 1
        mins, maxs = np.min(rxs, axis=0)-s, np.max(rxs, axis=0)+s
        
        plt.xlim([mins[0], maxs[0]])
        plt.ylim([mins[1], maxs[1]])
        if proj_3D:
            zlims = [mins[2], maxs[2]] if bs_pos is None else [np.min([mins[2], bs_pos[2]]),
                                                               np.max([mins[2], bs_pos[2]])]
            ax.axes.set_zlim3d(zlims)
    
    if equal_aspect: # often disrups the plot if in 3D.
        plt.axis('scaled')
    
    return fig, ax, cbar
