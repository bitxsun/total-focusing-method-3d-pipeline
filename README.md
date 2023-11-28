# total-focusing-method-pipeline-3d

Xiaoyu Sun @ University of Bristol, 2023-11-28

## Introduction
This repository contains python example scripts of 3D total focusing method (TFM) for pipeline inspections
Two TFM focusing approaches are involved, respectively the 1d interpolation and the neasrest point.

## Files
1. sdata-inpipe-wettissue50.amt is the example data acquired from a ring-shape, air-coupled ultrasonic array with a 40kHz centre frequency.
2. scode-script-tfm-3d-focallaw.py is the python script to generate focal law respectively for the direct ray path and the reflection ray path.
3. scode-script-tfm-3d-focallaw-plot.py is the python script to generate TFM image by using the focal law calculated from 'scode-script-tfm-3d-focallaw.py'
4. scode-script-tfm-3d-interp.py is the python script of using 1d interpolation approach to generate 3D TFM image.
5. scode-script-tfm-3d-nearest.py is the python script of using nearest point approach to generate 3D TFM image.
