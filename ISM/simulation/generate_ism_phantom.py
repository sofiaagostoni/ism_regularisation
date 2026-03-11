# %%
import os
from . import phantom_simulator as ph
import numpy as np
import torch
import CIL_winterschool.simulation.Tubulin_sim as st
import matplotlib.pyplot as plt


def generate_phantom(phantom_type, Nx, Ny, Nz, pxsizex):

        phantom = np.zeros([Nx, Nx], dtype=np.float32)

        if 'tubulin' in phantom_type or 'mixture' in phantom_type:
                tubulin_planar = st.tubSettings()
                tubulin_planar.xy_pixel_size = pxsizex
                tubulin_planar.xy_dimension = Nx
                tubulin_planar.xz_dimension = 1
                tubulin_planar.z_pixel =  1
                tubulin_planar.n_filament = 3
                tubulin_planar.radius_filament = pxsizex*1.0
                tubulin_planar.intensity_filament = [0.2, 1]
                phTub=np.zeros([Nz,Nx,Nx])
                for i in range(Nz):
                        phTub_planar= st.functionPhTub(tubulin_planar)
                        phTub_planar = np.swapaxes(phTub_planar, 2, 0)
                        phTub[i,:,:] = phTub_planar*(np.power(3,np.abs(i)))
                phantom += phTub.sum(axis=0)
                
        if 'balls' in phantom_type or 'mixture' in phantom_type:
                shape = (Nx, Nx)
                num = 10
                radius = 20
                value_range = (0.5, 1.0)
                smooth = True
                phantom += ph.generate_disk_phantom(shape, num, radius, value_range=value_range, smooth=smooth)
                
        if 'sparse' in phantom_type or 'mixture' in phantom_type:
                shape = (Nx, Nx)
                num = 20
                value_range = (0.5, 1.0)
                smooth = True
                phantom += ph.sparse_random_image(shape, num, value_range=value_range, margin=2)
                
        if 'nucleus' in phantom_type or 'mixture' in phantom_type:
                shape = (Nx, Nx)
                noOfGauss = int(300)
                value_range = (0.5, 0.8)
                std_range = (2, 6)
                ellipse_axes = (60, 30) # in nm
                phantom += ph.Nucleus(shape, noOfGauss, value_range, std_range, ellipse_axes, background_noise=0.05, seed=None)
                
        if 'membrane' in phantom_type or 'mixture' in phantom_type:
                shape = (Nx, Nx)
                num = 10
                value_range = (0.5, 0.8)
                radius=(65, 35)
                thickness=5.0
                noise_level=0.01
                blur_sigma=1.5
                smooth = True
                phantom += ph.generate_membrane_phantom(shape, radius, thickness, value_range, noise_level=noise_level, blur_sigma=blur_sigma)
                        
        if 'mitochondria' in phantom_type or 'mixture' in phantom_type:
                shape = (Nx, Nx)
                num = 20
                value_range = (0.8, 1.0)
                radius=(8, 6)
                thickness=1.0
                noise_level=0.01
                blur_sigma=1.0
                phantom += ph.generate_hollow_membrane_phantom(shape, num, radius, value_range, thickness, blur_sigma, noise_level)
                
                
        if 'argolight' in phantom_type:

                baseline = np.zeros((Nx, Nx))

                coord_l = np.arange(5, Nx-10, 13)

                spacing = np.arange(2, 11, 1)
                spacing_nm = (spacing - 1) * pxsizex

                coord_r = coord_l + spacing

                offset = 5
                line = np.arange(offset, Nx-offset, 1)

                for i in line:
                        for j in coord_l:
                                baseline[i, j] = 0.3 + torch.rand(1).item() * 0.3
                                # baseline[i, j] = 1                            
                        for j in coord_r:
                                baseline[i, j] = 0.3 + torch.rand(1).item() * 0.3
                                # baseline[i, j] = 1

                        


                # create tensor shaped image
                phantom = baseline
                
        phantom = torch.from_numpy(phantom).float()      
        phantom = phantom.unsqueeze(0).unsqueeze(0)
        return phantom
        
        