import matplotlib.pyplot as plt
import numpy as np


def plot_LoS_status(bs_location, user_locations, user_LoS, scat_size='auto'):
    LoS_map = {-1: ('r', 'No Path'), 0: ('b', 'NLoS'), 1: ('g', 'LoS')}
    
    plt.figure(dpi=300)
    
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


def plot_coverage(rxs, cov_map, dpi=200, figsize=(6,4), cbar_title=None, title=False,
                  scat_sz=.5, tx_pos=None, tx_ori=None, legend=False, lims=None,
                  proj_3D=False, equal_aspect=False, tight=True, cmap='viridis'):
    
    plt_params = {'cmap': cmap}
    if lims:
        plt_params['vmin'], plt_params['vmax'] = lims[0], lims[1]
    
    n = 3 if proj_3D else 2 # n coordinates to consider 2 = xy | 3 = xyz
    
    xyz = {'x': rxs[:,0], 'y': rxs[:,1]}
    if proj_3D:
        xyz['zs'] = rxs[:,2]
        
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize,
                           subplot_kw={'projection': '3d'} if proj_3D else {})
    
    im = plt.scatter(**xyz, c=cov_map, s=scat_sz, marker='s', **plt_params)

    cbar = plt.colorbar(im, label='Received Power [dBm][' if not cbar_title else cbar_title)
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # TX position
    
    if tx_pos is not None:
        ax.scatter(*tx_pos[:n], marker='P', c='r', label='TX')
    
    # TX orientation
    if tx_ori is not None and tx_pos is not None: # ori = [azi, el]
        # positive azimuths point left (like positive angles in a unit circle)    
        # positive elevations point up
        r = 30 # ref size of pointing direction
        tx_lookat = np.copy(tx_pos)
        tx_lookat[:2] += r * np.array([np.cos(tx_ori[2]), np.sin(tx_ori[2])]) # azimuth
        tx_lookat[2] += r * np.sin(tx_ori[1]) # elevation
        
        line_components = [[tx_pos[i], tx_lookat[i]] for i in range(n)]
        line = {key:val for key,val in zip(['xs', 'ys', 'zs'], line_components)}
        if n == 2:
            ax.plot(line_components[0], line_components[1], c='k', alpha=.5, zorder=3)
        else:
            ax.plot(**line, c='k', alpha=.5, zorder=3)
        # TODO: find arguments to plot the line in 3D
        # TODO: maintain scale in 3D plot
        
    if title:
        ax.set_title(title)
    
    if legend:
        plt.legend(loc='upper center', ncols=10, framealpha=.5)
    
    if tight:
        s = 1
        mins, maxs = np.min(rxs, axis=0)-s, np.max(rxs, axis=0)+s
        # TODO: change to set (simpler and to change space)
        if not proj_3D:
            plt.xlim([mins[0], maxs[0]])
            plt.ylim([mins[1], maxs[1]])
        else:
            ax.axes.set_xlim3d([mins[0], maxs[0]])
            ax.axes.set_ylim3d([mins[1], maxs[1]])
            if tx_pos is None:
                ax.axes.set_zlim3d([mins[2], maxs[2]])
            else:
                ax.axes.set_zlim3d([np.min([mins[2], tx_pos[2]]),
                                    np.max([mins[2], tx_pos[2]])])
    
    if equal_aspect and not proj_3D: # disrups the plot
        plt.axis('scaled')
    
    
    return fig, ax, cbar
