import matplotlib.pyplot as plt
import numpy as np

import Util.Global as g

def env(ues, aps, plot_c = False, save = False):
    
    fig, ax = plt.subplots(figsize = (10, 8))
    
    
    pos_aps = np.array(list(map(lambda ap: ap.position, aps)))
    
    pos_ues = np.array(list(map(lambda ue: ue.position, ues)))

    if plot_c:
        for ue in ues:
            for id_ap in ue.aps_connected:
                ax.plot([ue.position[0], aps[id_ap].position[0]], [ue.position[1], aps[id_ap].position[1]], '-k')
    
    ax.plot(pos_aps[:, 0], pos_aps[:, 1], 'or', ms = 10, label = f'{len(pos_aps)} APs')
    ax.plot(pos_ues[:, 0], pos_ues[:, 1], '*b', ms = 10, label = f'{len(pos_ues)} UEs')
    
    for i in range(len(pos_aps)):
        ax.annotate(i, (pos_aps[i, 0], pos_aps[i, 1]))
    
    for i in range(len(pos_ues)):
        ax.annotate(i, (pos_ues[i, 0], pos_ues[i, 1]))
        
    ax.set_xlim(g.ENV_X)
    ax.set_ylim(g.ENV_Y)
    ax.set_xlabel('meters', fontsize = 'x-large')
    ax.set_ylabel('meters', fontsize = 'x-large')
    ax.legend(loc = 'lower center', fontsize = 'x-large', framealpha = 1, shadow = True, ncol = 2, bbox_to_anchor =(.5, -.16), borderpad = .5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 'x-large')
    plt.tight_layout()
    
    if save:
        plt.savefig('Outputs/env.pdf')
    
    return fig, ax

        
def ue_params(ue, type = 'legacy'):
    fig, ax = None, None
    
    if type == 'legacy':
        fig, ax = plt.subplots(2, 2, figsize = (16, 10))
        ax[0, 0].plot(ue.gains, 'o-b', label = 'Path loss')
        ax[1, 0].plot(ue.dist_2D, 'o-b', label = 'Dist 2D')
        ax[0, 1].plot(ue.los_prob, 'o-b', label = 'Prob los')
        ax[1, 1].plot(ue.los, 'o-b', label = 'Bin los')
        
        ax[0, 0].tick_params(axis = 'x', which = 'major', labelsize = 'x-large')
        ax[0, 1].tick_params(axis = 'x', which = 'major', labelsize = 'x-large')
        ax[1, 0].tick_params(axis = 'x', which = 'major', labelsize = 'x-large')
        ax[1, 1].tick_params(axis = 'x', which = 'major', labelsize = 'x-large')
    
    elif type == 'pareto':
        fig, ax = plt.subplots(2, 2, figsize = (16, 10))
        idx_aps = np.argsort(ue.gains)        
        ax[0, 0].plot(ue.gains[idx_aps][::-1], 'o-b', label = 'Path loss')
        ax[0, 0].set_xticks(range(len(ue.gains)))
        ax[0, 0].set_xticklabels(idx_aps[::-1], rotation = 90, fontsize = 'small')
        
        ax[1, 0].plot(ue.dist_2D[idx_aps][::-1], 'o-b', label = 'Dist 2D')
        ax[1, 0].set_xticks(range(len(ue.dist_2D)))
        ax[1, 0].set_xticklabels(idx_aps[::-1], rotation = 90, fontsize = 'small')
        
        ax[0, 1].plot(ue.los_prob[idx_aps][::-1], 'o-b', label = 'Prob los')
        ax[0, 1].set_xticks(range(len(ue.los_prob)))
        ax[0, 1].set_xticklabels(idx_aps[::-1], rotation = 90, fontsize = 'small')
        
        ax[1, 1].plot(ue.los[idx_aps][::-1], 'o-b', label = 'Bin los')
        ax[1, 1].set_xticks(range(len(ue.los)))
        ax[1, 1].set_xticklabels(idx_aps[::-1], rotation = 90, fontsize = 'small')
            
            
        ax[0, 0].tick_params(axis = 'x', which = 'major', labelsize = 'small')
        ax[0, 1].tick_params(axis = 'x', which = 'major', labelsize = 'small')
        ax[1, 0].tick_params(axis = 'x', which = 'major', labelsize = 'small')
        ax[1, 1].tick_params(axis = 'x', which = 'major', labelsize = 'small')
        
        
    ax[0, 0].set_xlabel('$AP_{ID}$', fontsize = 'x-large')
    ax[0, 0].set_ylabel(r'$\frac{PL}{NOISE} + SF$', fontsize = 'x-large')
    ax[0, 0].legend(loc = 'upper right', fontsize = 'x-large', framealpha = .5, shadow = True, borderpad = .5)
    
    
    ax[1, 0].set_xlabel('$AP_{ID}$', fontsize = 'x-large')
    ax[1, 0].set_ylabel('meters', fontsize = 'x-large')
    ax[1, 0].legend(loc = 'upper right', fontsize = 'x-large', framealpha = .5, shadow = True, borderpad = .5)
    
    
    ax[0, 1].set_xlabel('$AP_{ID}$', fontsize = 'x-large')
    ax[0, 1].set_ylabel(r'Probability $\in [0, 1]$', fontsize = 'x-large')
    ax[0, 1].legend(loc = 'upper right', fontsize = 'x-large', framealpha = .5, shadow = True, borderpad = .5)
    
    
    ax[1, 1].set_xlabel('$AP_{ID}$', fontsize = 'x-large')
    ax[1, 1].set_ylabel(r'Probability $ \in \{0, 1\}$', fontsize = 'x-large')
    ax[1, 1].legend(loc = 'upper right', fontsize = 'x-large', framealpha = .5, shadow = True, borderpad = .5)
    
    
    ax[0, 0].tick_params(axis = 'y', which = 'major', labelsize = 'x-large')
    ax[0, 1].tick_params(axis = 'y', which = 'major', labelsize = 'x-large')
    ax[1, 0].tick_params(axis = 'y', which = 'major', labelsize = 'x-large')
    ax[1, 1].tick_params(axis = 'y', which = 'major', labelsize = 'x-large')
        
    plt.tight_layout()
    
    
    
    
    