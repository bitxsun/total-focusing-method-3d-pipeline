# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:43:32 2022
@author: xs16051
"""

import numpy as np
from scipy.io import loadmat, savemat
# import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = "browser" # "svg" "browser"


def create_cylinder(cy_radius, cy_length, cy_ang_gap):
    theta = np.linspace(0, 2 * np.pi, int(360 / cy_ang_gap))
    pipe_xc = np.tile(cy_radius * np.cos(theta), (len(cy_length), 1))
    pipe_yc = np.tile(cy_radius * np.sin(theta), (len(cy_length), 1))
    pipe_zc = np.tile(cy_length[:, np.newaxis], (1, len(theta)))
    return pipe_xc, pipe_yc, pipe_zc


#%% Load Array Geometry

tmp = loadmat('sarray-murata-alex-el64-dia150-sparse.mat')
el_xc = tmp['array'][0,0]['el_xc'][:,0].astype(float)
el_yc = tmp['array'][0,0]['el_yc'][:,0].astype(float)
el_zc = tmp['array'][0,0]['el_zc'][:,0].astype(float)

# Plot Element Geometry
# fig = go.Figure(data=go.Scatter(x=el_xc*1e2, y=el_yc*1e2, mode='markers', marker=dict(size=12,color='black')))
# fig.update_layout(scene=dict(xaxis_title='X (cm)', yaxis_title='Y (cm)'),
#                   font=dict(family="verdana", color="Black", size=18))
# fig.update_layout(yaxis=dict(scaleanchor="x",scaleratio=1,),
#                   xaxis=dict(constrain='domain'))
# fig.show()

#%% Create a pipe cylinder

# Imaging Area
x_size = 310 * 1e-3
y_size = 310 * 1e-3
z_sstt = 200 * 1e-3
z_send = 800 * 1e-3
p_size = 10 * 1e-3

pipe_radius = 150 * 1e-3
pipe_length = np.arange(0.0, (z_send+p_size), p_size)
pipe_scat_gap = 5 # Degree

pipe_xc, pipe_yc, pipe_zc = create_cylinder(pipe_radius, pipe_length, pipe_scat_gap)
pipe_xc = pipe_xc.flatten('C')
pipe_yc = pipe_yc.flatten('C')
pipe_zc = pipe_zc.flatten('C')

# Plot pipewall scatterers
# fig = go.Figure(data=go.Scatter3d(
#     x = pipe_zr * 1e2,
#     y = pipe_xr * 1e2,
#     z = pipe_yr * 1e2,
#     mode='markers',
#     marker=dict(size=3,color='red')
#     ))
# fig.update_layout(margin=dict(r=10, b=10, l=10, t=10))
# fig.update_layout(scene=dict(xaxis_title='Z (cm)', yaxis_title='X (cm)', zaxis_title='Y (cm)'),
#                   font=dict(family="verdana", color="Black", size=18))
# fig.show()

#%% TFM 3D

x = np.linspace(-x_size/2, x_size/2, int(x_size/p_size)+1)
y = np.linspace(-y_size/2, y_size/2, int(y_size/p_size)+1)
z = np.linspace(z_sstt, z_send, int((z_send-z_sstt)/p_size)+1)
[x_mg, y_mg, z_mg] = np.meshgrid(x, y, z)
x_mg = x_mg.flatten('C')
y_mg = y_mg.flatten('C')
z_mg = z_mg.flatten('C')

# Delay Law - Direct Ray Path
distd = np.sqrt((x_mg.reshape(-1, 1) - el_xc)**2 + 
                (y_mg.reshape(-1, 1) - el_yc)**2 + 
                (z_mg.reshape(-1, 1) - el_zc)**2)

# Delay Law - Element to Pipe Sactterers Ray Path
diste = np.sqrt((pipe_xc.reshape(-1, 1) - el_xc)**2 + 
                (pipe_yc.reshape(-1, 1) - el_yc)**2 + 
                (pipe_zc.reshape(-1, 1) - el_zc)**2)

# Delay Law - Pipe Sactterers to Pixel Ray Path
distp = np.sqrt((x_mg.reshape(-1, 1) - pipe_xc)**2 + 
                (y_mg.reshape(-1, 1) - pipe_yc)**2 + 
                (z_mg.reshape(-1, 1) - pipe_zc)**2)

#%% Focal Law

distr = np.zeros([len(x_mg), len(el_xc)])

for ii in range(0, len(el_xc), 1):
    distt = np.tile(diste[:, ii], (len(x_mg), 1))
    distr[:, ii] = np.min((distt + distp), axis=-1)
    del distt
    print(ii)
    
del diste, distp

focallaw = {'focallaw': {'focallaw_dir': distd,
                         'focallaw_ref': distr,
                         'array': tmp['array'][0,0],
                         'pixelx': x_mg,
                         'pixely': y_mg,
                         'pixelz': z_mg,
                         'pixelg': p_size,
                         'pipex': pipe_xc,
                         'pipey': pipe_yc,
                         'pipez': pipe_zc,
                         'pipe_radius': pipe_radius,
                         'pipe_scat_gap': pipe_scat_gap}}

#%% Save .mat File

array_type = 'murata-alex'
array_els = 64
savename = 'sdata-focal-law-' + array_type + '-' + str(array_els) + 'els-pipe' + str(int(pipe_radius*200)) + '-' + str(int(z_sstt*1e2)) + 'cm-to-' + str(int(z_send*1e2)) + 'cm.mat'

savemat(savename, focallaw)