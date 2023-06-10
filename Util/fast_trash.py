def min_dist(p):
    borders = [[g.ENV_X[0], p[1]], [g.ENV_X[1], p[1]], 
               [p[0], g.ENV_Y[0]], [p[0], g.ENV_Y[1]]]
    return min(list(map(lambda x: np.linalg.norm(x - p), borders)))

def fmt(x):
    return f'{x:.2f}'





gran = 2000
x = np.linspace(g.ENV_X[0], g.ENV_X[1], gran)
y = np.linspace(g.ENV_Y[0], g.ENV_Y[1], gran)
xm, ym = np.meshgrid(x, y)
r = np.vstack((xm.flatten(), ym.flatten())).T

z = np.array(list(map(min_dist, r)))
z = z.reshape((gran, gran))




dists = np.array([])
pos = np.array([500, 500, 100])


t = np.array([1400, 1500])




f = IntProgress(min = t[0], max = t[1], description = 'Progresso:')
display(f) # display the bar

for i in range(t[0], t[1]):
    pos_ues = np.array(list(map(lambda _: Factory.sample(type = 'uniform'), range(g.N_UES))))

    distance = 100
    pos_aps = Factory.topology(type = 'poisson', data = {'distance': distance}, shift = -3*distance, n = g.N_APS)
    pos_aps = np.array(list(filter(lambda p: g.ENV_X[0] <= p[0] <= g.ENV_X[1] and g.ENV_Y[0] <= p[1] <= g.ENV_Y[1], pos_aps)))

    dists = np.append(dists, np.array(list(map(min_dist, pos_aps[:, : 2]))))
    pos = np.row_stack((pos, pos_ues))


    # fig, ax = plt.subplots(2, 1, figsize = (10, 8))
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(2, 2, 1)

    ax.plot(pos_ues[:, 0], pos_ues[:, 1], '*b', ms = 8, label = f'{len(pos_ues)} UEs')
    ax.plot(pos_aps[:, 0], pos_aps[:, 1], 'or', ms = 10, label = f'{len(pos_aps)} APs')    
    ax.set_title(f'Simulation = {i+1}', fontsize = 'x-large')
    ax.set_xlim(g.ENV_X)
    ax.set_ylim(g.ENV_Y)
    # ax.set_title('APs and UEs positions: poisson disks', fontsize = 'x-large')
    ax.set_xlabel('x (meters)', fontsize = 'x-large')
    ax.set_ylabel('y (meters)', fontsize = 'x-large')
    ax.legend(loc = 'upper right', fontsize = 'x-large', framealpha = 1, shadow = True, borderpad = .5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 'x-large')






    ax = fig.add_subplot(2, 2, 2)

    cs = ax.contourf(xm, ym, z, levels = 30, alpha = 1, cmap = 'Greens')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 'x-large')

    fig.colorbar(cs, ax = ax)

    ax.clabel(cs, cs.levels[10::9], inline = False, fmt = fmt, fontsize = 12, colors = 'k')

    ax.set_xlabel('x (meters)', fontsize = 'x-large')
    ax.set_ylabel('y (meters)', fontsize = 'x-large')
    ax.set_title('Point-to-edge distance contours', fontsize = 'x-large')





    ax = fig.add_subplot(2, 2, 3)
    ax.hist(dists, bins = 50)

    ax.set_title('Point-to-edge distance histogram', fontsize = 'x-large')
    ax.set_xlabel('Point-to-edge distance', fontsize = 'x-large')
    ax.set_ylabel('Frequency', fontsize = 'x-large')
    ax.set_xlim([0 - 10, (g.ENV_X[1] - g.ENV_X[0]) / 2 + 10])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 'x-large')







    ax = fig.add_subplot(2, 2, 4, projection = '3d')
    hist, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1], bins=50)
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like(xpos)

    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()

    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax.set_title('APs positions: 3D histogram', fontsize = 'x-large')
    ax.set_xlabel('x (meters)', fontsize = 'x-large')
    ax.set_ylabel('y (meters)', fontsize = 'x-large')
    ax.set_zlabel('Frequency', fontsize = 'x-large')
    ax.view_init(50)




    plt.tight_layout()
    plt.savefig(f'Outputs/{i}_env.jpeg')
    plt.close(fig)

    f.value = i + 1
    
    
    
    
    
    
    
    
    
import imageio

total = 1500
frames = []
for i in range(total):
    image = imageio.v2.imread(f'Outputs/{i}_env.jpeg')
    frames.append(image)

imageio.mimsave(f'{total}_mygif.gif', frames, duration = 14) # d = 1000*t / |I|, t é o tempo em segundos que o gif vai ter


!gifsicle -i '1500_nbe.gif' -O3 --colors 64 -o '1500_nbe-64opt.gif'







import sys

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value))  for name, value in list(locals().items())), key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    
    
    
    
    
    
    
    
    
#     EXAMPLE:
#     data = {
#         0: {
#         'type': 'gaussiannpd',
#         'xbound': [1, 1.8],
#         'covs': .01,
#         'n' : 50,
#         'npd': 3
#         },

#         1: {
#         'type': 'gaussiannpd',
#         'xbound': [-2, -1],
#         'covs': .01,
#         'n' : 30,
#         'npd': 5
#         }
#     }
# f = lambda x: (x + 3) * (x - 2)**2 * (x + 1)**3
# xbounds = [-3, 2]
    @staticmethod
    def generate(f, xbound, data):
        res = np.array([[0, 0]]) # fake point just to concatenate
        
        for d in data.values():
            # generate n points per dot from gaussians
            if d['type'] == 'gaussiannpd':
                # data:
                # covs, xbound, n, npd (n per dots)
                cov = d['covs'] * np.array([[1, 0], [0, 1]])
                x = np.linspace(d['xbound'][0], d['xbound'][1], d['n'])
                y = np.array(list(map(lambda xx: f(xx), x)))            
                
                for x_val, y_val in zip(x, y):
                    r = np.random.multivariate_normal([x_val, y_val], cov, d['npd'])
                    res = np.concatenate((res, r), axis = 0)
        
        return res[1:, :] # remove fake point
    
    
    
    
    
    
# Generate UEs moving

from Classes.Brownian import Brownian


dist_bp = aux.breakpoint_distance()
noise_var = aux.noise_power()
aps = Factory.topologyAP(type = 'poisson', data = {'distance': 50}, shift = 100)
n_ues = 3
ues = [*map(lambda _: Factory.sampleUE(shift = 20), range(n_ues))]


x = [*map(lambda ue: Brownian(float(ue.position[0])), ues)]
y = [*map(lambda ue: Brownian(float(ue.position[1])), ues)]

n = 1000
wx = [*map(lambda xx: xx.gen_random_walk(n), x)]
wy = [*map(lambda yy: yy.gen_random_walk(n), y)]


for i in range(n):
    for ue in ues:
        for id_ap, ap in enumerate(aps):
            ue.dist_3D[id_ap] = aux.calc_distance(ue.position, ap.position)
            ue.dist_2D[id_ap] = aux.calc_distance(ue.position[:-1], ap.position[:-1])
            ue.los_prob[id_ap] = aux.los_probability('UMI', ue.dist_2D[id_ap], threshold2D = g.UMI_THRESHOLD_DIST)
            # calcular o ganho da 'ue' para a 'ap'
            _, _, pl, los = aux.path_loss('UMI', ue.los_prob[id_ap], ue.dist_2D[id_ap], ue.dist_3D[id_ap], dist_bp, noise_var)
            ue.gains[id_ap] = pl
            ue.los[id_ap] = los

    cluster(ues, aps, type = 'dist')

    fig, ax = show.env(ues, aps, plot_c = True)

    plt.savefig(f'Outputs/{i}_env.jpeg')
    plt.close(fig)
    # ue.position = np.random.multivariate_normal(ue.position, 50 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    for j, ue in enumerate(ues):
        ue.position[0] = wx[j][i]
        ue.position[1] = wy[j][i]
        
        
        
import imageio

total = 1000
frames = []
for i in range(total):
    image = imageio.v2.imread(f'Outputs/{i}_env.jpeg')
    frames.append(image)

imageio.mimsave(f'{total}_mygif.gif', frames, duration = 20) # d = 1000*t / |I|, t é o tempo em segundos que o gif vai ter



!gifsicle -i '1000_mygif.gif' -O3 --colors 8 -o '1000_mygif_8opt.gif'