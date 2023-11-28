# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 08:28:09 2023

@author: xs16051
"""

# import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.signal import hilbert, butter, lfilter
from scipy.interpolate import interp1d
from scipy.io import loadmat
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='bandpass')
    y = lfilter(b, a, data, axis)
    return y

#%% Load Focallaw

focallaw = loadmat('sdata-focal-law-murata-alex-64els-pipe30-20cm-to-80cm.mat')
distd = focallaw['focallaw']['focallaw_dir'][0,0]
distr = focallaw['focallaw']['focallaw_ref'][0,0]
el_xc = focallaw['focallaw']['array'][0,0]['el_xc'][0,0][:,0]
el_yc = focallaw['focallaw']['array'][0,0]['el_yc'][0,0][:,0]
el_zc = focallaw['focallaw']['array'][0,0]['el_zc'][0,0][:,0]
x_mg = focallaw['focallaw']['pixelx'][0,0][0,:]
y_mg = focallaw['focallaw']['pixely'][0,0][0,:]
z_mg = focallaw['focallaw']['pixelz'][0,0][0,:]
pipe_xc = focallaw['focallaw']['pipex'][0,0][0,:]
pipe_yc = focallaw['focallaw']['pipey'][0,0][0,:]
pipe_zc = focallaw['focallaw']['pipez'][0,0][0,:]
pipe_rd = focallaw['focallaw']['pipe_radius'][0,0][0,0]

del focallaw

#%% Load Data

tmp = loadmat('sdata-wood-big.mat')
timx = tmp['exp_data']['time'][0,0][0,:]
time_data = tmp['exp_data']['time_data'][0,0]
tx = (tmp['exp_data']['tx'][0,0][0,:]-1).astype(int)
rx = (tmp['exp_data']['rx'][0,0][0,:]-1).astype(int)
el_xc = tmp['exp_data']['array'][0,0]['el_xc'][0,0][0,:].astype(float)
el_yc = tmp['exp_data']['array'][0,0]['el_yc'][0,0][0,:].astype(float)
el_zc = tmp['exp_data']['array'][0,0]['el_zc'][0,0][0,:].astype(float)
fs = 1 / (timx[1] - timx[0])
ph_velocity = 343
imaging_type = 'interp'
db_scale = -20

#%% Apply Butter bandpass filter

lowcut = 20e3  # Low cutoff frequency of the filter, Hz
highcut = 60e3  # High cutoff frequency of the filter, Hz
data_flt = butter_bandpass_filter(time_data, lowcut, highcut, fs)

#%% TFM 3D

delay = distr / ph_velocity

tt = time.time()

II = np.zeros(x_mg.shape, dtype=complex)


if imaging_type == 'interp':
    for ii in range(0, time_data.shape[1], 1):
        itp = interp1d(timx, hilbert(data_flt[:, ii]), kind='linear', fill_value=0)
        II += itp((delay[:, tx[ii]] + delay[:, rx[ii]]))
elif imaging_type == 'nearest':
    for ii in range(0, time_data.shape[1], 1):
        idx = np.round((delay[:, tx[ii]] + delay[:, rx[ii]]) * fs)
        II += hilbert(data_flt[:, ii])[idx.astype(int)]
else:
    print('Unrecognised Imaging Input!')
II_db = 20 * np.log10(abs(II)/np.max(abs(II)))

print('TFM runs: ' + str(time.time()-tt) + ' seconds')

#%% Plot TFM

pipe_theta = np.linspace(0, 2*np.pi, 100)
pipe_zc = np.linspace(0, 1, 50)
pipe_ang, pipe_zc = np.meshgrid(pipe_theta, pipe_zc)
vector_sin = np.vectorize(np.sin)
vector_cos = np.vectorize(np.cos)
pipe_xc = pipe_rd * vector_cos(pipe_ang)
pipe_yc = pipe_rd * vector_sin(pipe_ang)

pipe_color = [[0, 'red'],
              [1, 'red']]

pio.renderers.default = "browser" # "svg" "browser"
fig = go.Figure(data=go.Isosurface(
    x = z_mg.flatten() * 1e2,
    y = x_mg.flatten() * 1e2,
    z = y_mg.flatten() * 1e2,
    value=II_db,
    isomin=db_scale,
    isomax=0,
    caps=dict(x_show=False, y_show=False),
    colorscale='jet',
    showscale=False,
    surface_count=6,
    opacity=0.3
    ))
fig.add_trace(go.Scatter3d(
    x = el_zc * 1e2,
    y = el_xc * 1e2,
    z = el_yc * 1e2,
    mode='markers',
    marker=dict(
        size=10)
    ))
fig.add_trace(go.Surface(
    x = pipe_zc * 1e2,
    y = pipe_xc * 1e2,
    z = pipe_yc * 1e2,
    colorscale = pipe_color,
    showscale=False,
    opacity=0.1
    ))
camera = dict(eye=dict(x=1.5, y=2.5, z=0.6))
fig.update_layout(scene_camera = camera)
fig.update_layout(scene_aspectmode='data')
fig.update_layout(margin=dict(r=10, b=10, l=10, t=10))
fig.update_layout(scene=dict(xaxis_title='Z (cm)', yaxis_title='X (cm)', zaxis_title='Y (cm)'),
                  font=dict(family="verdana", color="Black", size=18))
fig.show()
# fig.write_html("images/brick1_3d_saft.html")