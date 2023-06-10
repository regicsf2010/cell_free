import numpy as np

import Util.Global as g
from Classes.PoissonDisc import PoissonDisc
from Classes.AP import AP
from Classes.UE import UE

class Factory:
    
    # Not required since all methods are static
    def __init__():
        pass
    
    
    ############# FOR USERS #############
    @staticmethod
    def sampleUE(shift: int = 0, dx: [] = g.ENV_X, dy: [] = g.ENV_Y, h_ue = g.H_UE):
        pos_xy = Factory.sample(shift, dx, dy)
        return UE(pos_xy[0], pos_xy[1], h_ue)
    
    @staticmethod
    def sampleAP(shift: int = 0, dx: [] = g.ENV_X, dy: [] = g.ENV_Y, h_ap = g.H_AP):
        pos_xy = Factory.sample(shft, dx, dy, g_ap)
        return AP(pos_xy[0], pos_xy[1], h_ap)
    
    @staticmethod
    def topologyAP(data, type: str = 'grid', n: int = g.N_APS, shift: int = 0, dx: [] = g.ENV_X, dy: [] = g.ENV_Y, h_ap = g.H_AP):
        pos_xy = Factory.topology(data, type, n, shift, dx, dy)
        return list(map(lambda i: AP(pos_xy[i, 0], pos_xy[i, 1], h_ap), range(0, n)))
    
    @staticmethod
    def topologyUE(data, type: str = 'grid', n: int = g.N_UES, shift: int = 0, dx: [] = g.ENV_X, dy: [] = g.ENV_Y, h_ue = g.H_UE):
        pos_xy = Factory.topology(data, type, n, shift, dx, dy)
        return list(map(lambda i: UE(pos_xy[i, 0], pos_xy[i, 1], h_ue), range(0, n)))
    
    ######################################
    
    
    
    
    
    
    
    
    
    ############# INTERNAL PURPOSE (BUT IT CAN BE USED BY A USER AS WELL) #############
    # Uniform random point generator over a pre-defined space ([dx, dy]) with or without a shift
    @staticmethod
    def sample(shift: int = 0, dx: [] = g.ENV_X, dy: [] = g.ENV_Y):
        return np.array([np.random.uniform(dx[0] + shift, dx[1] - shift), 
                         np.random.uniform(dy[0] + shift, dy[1] - shift)])
    
    @staticmethod
    def topology(data, type: str = 'grid', n: int = 0, shift: int = 0, dx: [] = g.ENV_X, dy: [] = g.ENV_Y):
        if type == 'grid':
            
            if data['matrix'][0] * data['matrix'][1] != n:
                print('Not possible to build matrix.')
                return None
            
            x = np.linspace(dx[0] + shift, dx[1] - shift, data['matrix'][1])
            y = np.linspace(dy[0] + shift, dy[1] - shift, data['matrix'][0])
            
            pos = np.vstack((np.tile(x, data['matrix'][0]),
                             np.tile(y, data['matrix'][1]))).T
            
        elif type == 'spiral':
            n_spirals = data['n_spirals']
            
            if n % n_spirals != 0:
                return None
            
            center = np.array([(dx[1] - dx[0]), (dy[1] - dy[0])]) / 2
            
            x = np.linspace(dx[0] + shift, dx[1] - center[0] - shift, n_spirals)
            
            points = []
            for x_val in x:
                rad = np.random.uniform(0, 2 * np.pi) # random initial rad
                n_points_per_circle = int(n / n_spirals)
                
                for _ in range(n_points_per_circle):
                    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
                    
                    p = np.dot(rot, np.array([x_val, 0]))
                    p = center + p
                    points.append([p[0], p[1]])
                    
                    rad = (rad + (2 * np.pi / n_points_per_circle)) % (2 * np.pi)
            
            pos = np.array(points)
            
        elif type == 'min_dist':
            distance = data['distance']
            
            pos = np.array([np.random.uniform(dx[0] + shift, dx[1] - shift),
                               np.random.uniform(dy[0] + shift, dy[1] - shift)])
            
            c = 2
            while c <= n:
                candidate = np.array([np.random.uniform(dx[0] + shift, dx[1] - shift),
                                      np.random.uniform(dy[0] + shift, dy[1] - shift)])
                
                min_distance = np.min(list(map(lambda x: np.linalg.norm(x - candidate), pos)))
                
                if min_distance > distance:
                    pos = np.row_stack((pos, candidate))
                    c += 1                            

        elif type == 'cluster':
            n_clusters = data['n_clusters']
            radius = data['radius']
            overlap = data['overlap']
            
            points_per_cluster = int(n / n_clusters)
            
            pos = []
            
            if overlap:
                clusters_x = np.random.uniform(dx[0] + shift, dx[1] - shift, n_clusters)
                clusters_y = np.random.uniform(dy[0] + shift, dy[1] - shift, n_clusters)
                
                
                for c in zip(clusters_x, clusters_y):
                    r = np.random.multivariate_normal(c, radius * np.array([[1, 0], [0, 1]]), points_per_cluster)
                    pos.extend(r)
            
            else:
                pass
            
            
        elif type == 'poisson':
            pd = PoissonDisc(width = dx[1], height = dy[1], r = data['distance'])
            
            # sample points without considering the shift value
            # not that we don't have control on how many points will be generated
            pos = pd.sample() 
            
            # convert to the original dx and dy with shift value
            min_x, max_x = np.min(pos[:, 0]), np.max(pos[:, 0])
            min_y, max_y = np.min(pos[:, 1]), np.max(pos[:, 1])
            
            pos[:, 0] = np.interp(pos[:, 0], [min_x, max_x], [dx[0] + shift, dx[1] - shift])
            pos[:, 1] = np.interp(pos[:, 1], [min_y, max_y], [dy[0] + shift, dy[1] - shift])            
            
            pos = np.random.permutation(pos)
        
        
        n = np.min([n, len(pos)]) # because we don't have control over 'poisson method'.
        
        # Return just the first n points required (only required for poisson method)
        # For other methods, n and len(pos) will be exact the same value
        return pos[:n, :] 