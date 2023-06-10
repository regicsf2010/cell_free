import Util.Global as g
import numpy as np

class UE:

    def __init__(self, x: float = 0, y: float = 0, z: float = 0, min_aps = g.MIN_APS_PER_UE, max_aps = g.MAX_APS_PER_UE):
        self.position = np.array([x, y, z])
        
        self.min_aps = min_aps
        self.max_aps = max_aps
        
        self.ap_master = g.NO_AP_MASTER
        self.aps_connected = []
        
        self.dist_2D = np.zeros(g.N_APS)
        self.dist_3D = np.zeros(g.N_APS)
        self.los_prob = np.zeros(g.N_APS)
        self.los = np.zeros(g.N_APS)
        self.gains = np.zeros(g.N_APS) # path losses
        
        
    def set_los_prob(los_prob):
        self.los_prob = los_prob
        
    def reset_aps(self):
        self.aps_connected = []
        self.ap_master = g.NO_AP_MASTER
    
    
    def __str__(self):
        return f'( {self.x:7.2f} m, {self.y:7.2f} m, {self.z:7.2f} m )'
    