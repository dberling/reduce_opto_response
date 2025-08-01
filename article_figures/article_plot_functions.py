import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.cm as cm

def label_subplots_ABC(fig, axs, x_shift, y_shift, label_shift=0, **text_kws):
    labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    if type(x_shift) != list():
        x_shift = [x_shift] * len(axs)
    if type(y_shift) != list():
        y_shift = [y_shift] * len(axs)
    for i, ax in enumerate(axs):
        # Get the position of the subplot
        pos = ax.get_position()  # returns (left, bottom, width, height)
        
        # Annotate at the upper-left corner of each subplot
        fig.text(pos.x0 + x_shift[i], pos.y1 + y_shift[i], labels[i+label_shift], **text_kws)

def get_map_patt_id_to_x_y(xy_max, dxy):
    xys = np.arange(-1*xy_max, xy_max, dxy)
    xs, ys = [arr.flatten() for arr in np.meshgrid(xys, xys)]
    patt_ids = np.arange(0,len(xs))
    mapper = dict()
    for patt_id, x, y in zip(patt_ids, xs, ys):
        mapper[patt_id] = (x, y)
    return mapper


def map_patt_id_2_xy(df, xy_max, dxy, rotate):
    mapper = get_map_patt_id_to_x_y(xy_max, dxy)
    df['x'] = [mapper[patt_id][0] for patt_id in df['patt_id']]
    df['y'] = [mapper[patt_id][1] for patt_id in df['patt_id']] 
    if rotate:
            df['x'] = df['x'] * (-1) #+ dxy
            df['y'] = df['y'] * (-1) #+ dxy
    return df
    
def subplot(df, ax, APCmax, cmap, shift_x=0, shift_y=0, APCmin=0):
    pivot = df.pivot(index='x', columns='y', values='APC')
    xx, yy = np.meshgrid(pivot.index, pivot.columns, indexing='ij')
    xx += shift_x
    yy += shift_y
    mappable = ax.pcolormesh(xx, yy, pivot.values, shading='nearest', cmap=cmap, vmin=APCmin, vmax=APCmax)
    return ax, mappable

def spatial_activation_plot(ax, df, APCmax, cell_topview_simtree, morph_lw, shift_x, sb_draw=1, sb_width=2, label_fiber=False):
    print("APCmax: ",df.APC.max())
    # prepare cmap:
    cmap = LinearSegmentedColormap.from_list('custom_reds', ['white', 'red'], N=256)
    # plot activation
    ax, mappable = subplot(df, ax, APCmax=APCmax, cmap=cmap, shift_x=shift_x)
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('equal')
    ax.set_ylim(-126,126)
    
    # Create unfilled circle for fiber
    circle_center = (-100, -100)  # Circle center in data coordinates
    circle_radius = 25
    arrow_length = 50
    facecolor = LinearSegmentedColormap.from_list('custom_blues', ['white', 'blue'], N=256)(0.8)
    circle = patches.Circle(circle_center, circle_radius, edgecolor='black', facecolor=facecolor, linewidth=2)
    ax.add_patch(circle)
    # and arrows with movement directions
    # Arrow pointing UP
    ax.arrow(circle_center[0], circle_center[1] + circle_radius, 0, arrow_length, 
             head_width=5, head_length=10, fc='black', ec='black')
    # Arrow pointing RIGHT
    ax.arrow(circle_center[0] + circle_radius, circle_center[1], arrow_length, 0, 
             head_width=5, head_length=10, fc='black', ec='black')
    # label
    if label_fiber:
        ax.text(circle_center[0], circle_center[1] - circle_radius, 'optical\nfiber',
                ha='left', va='top', fontsize=8)
    
    # Plot top view of cell
    cell_topview_simtree.plot2DMorphology(
        ax=ax,
        use_radius=True,
        plotargs={'c': 'k', 'lw': morph_lw, 'alpha':1},
        sb_width=sb_width,
        sb_draw=sb_draw
    )
    return ax, mappable

def spatial_activation_diff_plot(ax, df_full, df_reduced, APCmin, APCmax, cell_topview_simtree, morph_lw, sb_draw=1, sb_width=2, label_fiber=False, relative=False):
    df = df_reduced[['x','y','APC']].merge(df_full[['x','y','APC']], on=['x','y'])
    if relative:
        df['APC'] = (df.APC_x - df.APC_y) / df.APC_y
    else:
        df['APC'] = (df.APC_x - df.APC_y)
        #df.loc[df.APC == 0]['APC'] = np.nan
    
    print("APCmax: ",df.APC.max())
    print("APCmin: ",df.APC.min())
    # prepare cmap:
    #cmap = 'PuOr'
    original_cmap = plt.cm.get_cmap('PuOr', 256)
    colors = original_cmap(np.linspace(0, 1, 256))
    colors[128] = [1, 1, 1, 1]  # RGBA for white
    cmap = LinearSegmentedColormap.from_list('PuOr_white_center', colors)
    # plot activation
    ax, mappable = subplot(df, ax, APCmin=APCmin, APCmax=APCmax, cmap=cmap)
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('equal')
    ax.set_ylim(-126,126)
    
    # Create unfilled circle for fiber
    circle_center = (-100, -100)  # Circle center in data coordinates
    circle_radius = 25
    arrow_length = 50
    facecolor = LinearSegmentedColormap.from_list('custom_blues', ['white', 'blue'], N=256)(0.8)
    circle = patches.Circle(circle_center, circle_radius, edgecolor='black', facecolor=facecolor, linewidth=2)
    ax.add_patch(circle)
    # and arrows with movement directions
    # Arrow pointing UP
    ax.arrow(circle_center[0], circle_center[1] + circle_radius, 0, arrow_length, 
             head_width=5, head_length=10, fc='black', ec='black')
    # Arrow pointing RIGHT
    ax.arrow(circle_center[0] + circle_radius, circle_center[1], arrow_length, 0, 
             head_width=5, head_length=10, fc='black', ec='black')
    # label
    if label_fiber:
        ax.text(circle_center[0], circle_center[1] - circle_radius, 'optical\nfiber',
                ha='left', va='top', fontsize=8)
    
    # Plot top view of cell
    cell_topview_simtree.plot2DMorphology(
        ax=ax,
        use_radius=True,
        plotargs={'c': 'k', 'lw': morph_lw, 'alpha':1},
        sb_width=sb_width,
        sb_draw=sb_draw
    )
    return ax, mappable

    
def plot_morphology_with_light(ax, simtree, marklocs, locargs, morph_lw, light_prof_xx_zz_I, inset=False, inset_data=None, sb_width=5.0, sb_draw=True, cax=None, cbar_kws=dict(), plotargs={'c': 'k', 'alpha':1}, cs=None, cmap=None):
    if 'lw' not in plotargs.keys():
        plotargs['lw'] = morph_lw
    custom_cmap = LinearSegmentedColormap.from_list('custom_blues', ['white', 'blue'], N=256)
    # plot light profile
    mappable = ax.pcolormesh(light_prof_xx_zz_I[0],light_prof_xx_zz_I[1],light_prof_xx_zz_I[2], cmap=custom_cmap)
    # plot cell morphology
    simtree.plot2DMorphology(
        ax=ax,
        plotargs=plotargs,
        marklocs=marklocs,
        locargs=locargs,
        draw_soma_circle=False,
        sb_width=sb_width,
        sb_draw=sb_draw,
        cmap=cmap
    )
    if inset:
        # create inset with conductance
        inset_ax = inset_axes(ax, width="20%", height="20%", loc="center right")  # Adjust size and location
        inset_ax.plot(inset_data['time [ms]'], inset_data['g_soma'], label='somatic', c='black',ls='dashed')
        inset_ax.plot(inset_data['time [ms]'], inset_data['g_apical'], label='apical', c='g')
        inset_ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.05))
        inset_ax.spines["top"].set_visible(False)
        inset_ax.spines["right"].set_visible(False)
        inset_ax.spines["left"].set_visible(False)
        inset_ax.set_yticks([])
        inset_ax.set_xticks([0,25,50])
        inset_ax.set_xticklabels(['0','','50'])
        inset_ax.set_xlabel('time [ms]')
    if cax != None:
        plt.colorbar(mappable, cax=cax, **cbar_kws)
    return ax, mappable
