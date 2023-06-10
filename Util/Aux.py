import numpy as np

import Util.Global as g


def breakpoint_distance(h_AP = g.H_AP, h_UE = g.H_UE, fc = g.FREQ_Hz, c = g.SPEED_OF_LIGHT, h_E = 0):
    return 4 * (h_AP - h_E) * (h_UE - h_E) * (fc / c)
        
def noise_power(bw = g.BW, noise = g.NOISE): # noiseVariance_dBm = -170 + 10*log10(BW_Hz) + noiseFigure_dB;    
    return -170 + 10 * np.log10(bw) + noise


def calc_distance(ue, ap):
    return np.linalg.norm(ue - ap)

# dist2D = distance without height: dist(x, y) = np.sqrt(np.sum((x - y)**2))
def los_probability(scenario, dist2D, threshold2D):
    if scenario == 'UMI': # threshold = 18 meters
        if dist2D <= threshold2D:
            return 1.0
        else:
            return (threshold2D / dist2D) + np.exp(- dist2D / (2 * threshold2D)) * (1 - (threshold2D / dist2D))



def path_loss(scenario, los_prob, dist_2D, dist_3D, dist_bp, noise_var, h_AP = g.H_AP, h_UE = g.H_UE, fc = g.FREQ_GHz):
    los = 1
    if scenario == 'UMI':        
        if np.random.uniform() <= los_prob:
            if dist_2D <= dist_bp:
                pl = 32.4 + 21 * np.log10(dist_3D) + 20 * np.log10(fc)
            else:
                pl = 32.4 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) - 9.5 * np.log10(dist_bp**2 + (h_AP - h_UE)**2)
            
            shadow_fading = np.random.normal(0, g.STD_LOS) # move to outside
            
        else:
            los = 0
            pl_nlos = 35.3 * np.log10(dist_3D) + 22.4 + 21.3 * np.log10(fc) - 0.3 * (h_UE - 1.5)
            pl_los = path_loss(scenario, 1, dist_2D, dist_3D, dist_bp, noise_var, h_AP, h_UE, fc)[0]
            pl = np.max([pl_los, pl_nlos])
            
            shadow_fading = np.random.normal(0, g.STD_NLOS) # move to outside
    
    # log(in/out / noise) = 
    pl_over_noise = -pl - noise_var
    
    return pl, pl_over_noise, pl_over_noise + shadow_fading, los