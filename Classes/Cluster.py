import numpy as np

import Util.Global as g

class Cluster:
    
    def __init__(self):
        pass

    def cluster(self, ues, aps, type = 'uac'):
    
        # reset previous connections
        for ue in ues:
            ue.reset_aps()
        for ap in aps:
            ap.reset_ues()

        # cluster ues based on a clusterization method
        if type == 'pathloss':
            for id, _ in enumerate(ues):
                ids_aps = np.argsort(ues[id].gains)[::-1]

                c_conn = 0
                c_ap = 0
                while c_conn < ues[id].min_aps and c_ap < g.N_APS:
                    if len(aps[ids_aps[c_ap]].ues_connected) < aps[ids_aps[c_ap]].max_ues:
                        ues[id].aps_connected.append(ids_aps[c_ap])
                        aps[ids_aps[c_ap]].ues_connected.append(id)
                        c_conn = c_conn + 1
                        if c_conn == 1:
                            ues[id].ap_master = ids_aps[c_ap]

                    c_ap = c_ap + 1

        elif type == 'dist':
            for id, _ in enumerate(ues):
                ids_aps = np.argsort(ues[id].dist_2D)

                c_conn = 0
                c_ap = 0
                while c_conn < ues[id].min_aps and c_ap < g.N_APS:
                    if len(aps[ids_aps[c_ap]].ues_connected) < aps[ids_aps[c_ap]].max_ues:
                        ues[id].aps_connected.append(ids_aps[c_ap])
                        aps[ids_aps[c_ap]].ues_connected.append(id)
                        c_conn = c_conn + 1
                        if c_conn == 1:
                            ues[id].ap_master = ids_aps[c_ap]

                    c_ap = c_ap + 1