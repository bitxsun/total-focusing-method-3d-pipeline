# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:43:32 2022
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


#%% Load data

tmp = loadmat('sdata-inpipe-wettissue50.mat')
timx = tmp['exp_data']['time'][0,0][:,0]
time_data = tmp['exp_data']['time_data'][0,0]
tx = (tmp['exp_data']['tx'][0,0][0,:]-1).astype(int)
rx = (tmp['exp_data']['rx'][0,0][0,:]-1).astype(int)
el_xc = tmp['exp_data']['array'][0,0]['el_xc'][0,0][:,0].astype(float)
el_yc = tmp['exp_data']['array'][0,0]['el_yc'][0,0][:,0].astype(float)
el_zc = tmp['exp_data']['array'][0,0]['el_zc'][0,0][:,0].astype(float)
fs = 1 / (timx[1] - timx[0])
ph_velocity = 343
db_scale = -6

#%% Apply Butter bandpass filter

lowcut = 20e3  # Low cutoff frequency of the filter, Hz
highcut = 60e3  # High cutoff frequency of the filter, Hz
data_flt = butter_bandpass_filter(time_data, lowcut, highcut, fs)

# plt.figure(figsize=(8, 6))
# plt.plot(timx*1e3, time_data[:, 0], label='Original signal')
# plt.plot(timx*1e3, data_flt[:, 0], label='Filtered signal')
# plt.grid(color='0.7', linestyle=':', linewidth=0.5)
# plt.xlim(0, 10)
# plt.xlabel('Time (ms)')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.tight_layout()
# plt.show()

#%% TFM 3D

tt = time.time()

x_size = 310 * 1e-3
y_size = 310 * 1e-3
z_sstt = 200 * 1e-3
z_send = 800 * 1e-3
p_size = 10 * 1e-3

x = np.linspace(-x_size/2, x_size/2, int(x_size/p_size)+1)
y = np.linspace(-y_size/2, y_size/2, int(y_size/p_size)+1)
z = np.linspace(z_sstt, z_send, int((z_send-z_sstt)/p_size)+1)
[x_mg, y_mg, z_mg] = np.meshgrid(x, y, z)

# Calculate delay law
delay = np.sqrt((x_mg.flatten().reshape(-1, 1) - el_xc)**2 + 
                (y_mg.flatten().reshape(-1, 1) - el_yc)**2 + 
                (z_mg.flatten().reshape(-1, 1) - el_zc)**2) / ph_velocity

# Generate TFM image
II = np.zeros(x_mg.flatten().shape, dtype=complex)
for ii in range(0, time_data.shape[1], 1):
    itp = interp1d(timx, hilbert(data_flt[:, ii]), kind='linear', fill_value=0)
    II += itp((delay[:, tx[ii]] + delay[:, rx[ii]]))

II_db = 20 * np.log10(abs(II)/np.max(abs(II)))

print('TFM runs: ' + str(time.time()-tt) + ' seconds')

#%% Plot TFM

pipe_radius = 150 * 1e-3
pipe_theta = np.linspace(0, 2*np.pi, 100)
pipe_zc = np.linspace(0, z_send, 50)
pipe_ang, pipe_zc = np.meshgrid(pipe_theta, pipe_zc)
vector_sin = np.vectorize(np.sin)
vector_cos = np.vectorize(np.cos)
pipe_xc = pipe_radius * vector_cos(pipe_ang)
pipe_yc = pipe_radius * vector_sin(pipe_ang)

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