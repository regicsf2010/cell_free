import Util.Global as g
import numpy as np

class AP:

    def __init__(self, x: float = 0, y: float = 0, z: float = 0, max_ues: int = g.MAX_UES_PER_AP):
        self.position = np.array([x, y, z])
        
        self.max_ues = max_ues
        self.ues_connected = [] 
             
    
    def reset_ues(self):
        self.ues_connected = []
        
        
    def __str__(self):
        return f'( {self.x:7.2f} m, {self.y:7.2f} m, {self.z:7.2f} m )'
    