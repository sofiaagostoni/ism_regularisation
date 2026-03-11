
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import deepinv as dinv
from opt_functions import * 

# Librerie specifiche ISM
import ISM.dataio.mcs as mcs
import ISM.analysis.APR_lib as apr
import ISM.analysis.Graph_lib as gr
import ISM.simulation.PSF_sim as sim
import ISM.analysis.FocusISM_lib as fism

from s2ism import  psf_estimator as est
from s2ism import  s2ism  as amd

# Configurazione Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def save_fromh5_totorch(path, file, Nz, name, exwl, emwl):
        
        # path = r'Data_results/Real_data'       #es: \\iitfsvge101.iit.local\mms\Data MMS server\STED-ISM\AxialDeconvolution\Convallaria\C

        # file = r'08-Tubulin-Mo-CF488A-03-03-2026-18-17-56.h5'

        # Nz = 2

        # name = "08_tubulin"
        
        dset, PSF, meta = load_real_data(path, file, Nz)

        data = {
        "measurment": dset,
        "PSF": PSF,
        "metadati": meta,
        "exwl": exwl,
        "emwl": emwl
        }

        torch.save(data, f'Data_results/Real_data/{name}_data.pth')

def load_real_data(path, file, Nz):

    # Configurazione Grafica
    # plt.rcParams.update({'ps.fonttype': 42, 'pdf.fonttype': 42, 
    #                     'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})
    plt.close('all')

    fullpath = os.path.join(path, file)

    data, meta = mcs.load(fullpath)
    print(f"data shape {data.shape}")
    dset = data
    # dset = dset.sum(1)
    if dset.shape[1] > 1:
        dset = dset[:,int(dset.shape[1]/2) - 1: int(dset.shape[1]/2)]
        
    dset = np.squeeze(dset)
    dset = dset.sum(-2)
    print(f"data shape {dset.shape}")


    # Generazione immagini standard (Closed/Open Pinhole)
    
    exPar = sim.simSettings()
    exPar.wl = 488
    exPar.mask_sampl = 101

    emPar = exPar.copy()
    emPar.wl = 520
    Nx = meta.nx
    Ny = meta.ny

    grid = sim.GridParameters()
    grid.pxsizex = meta.dx*1e3

    grid.Nz = Nz

    PSF, detPSF, exPSF = est.psf_estimator_from_data(dset, exPar, emPar, grid, z_out_of_focus = 700)
    spad_size = grid.spad_size() / emPar.airy_unit

    PSF = torch.from_numpy(PSF).float().to(device)
    PSF = PSF.permute(3, 0, 1, 2)    

    dset = torch.from_numpy(dset).float().to(device)
    dset = dset.unsqueeze(0)
    dset = dset.permute(3, 0, 1, 2)  
    
    if Nz == 1: 
        PSF = PSF / PSF.sum()
        
    if Nz == 2:
        
        PSF[:,0:1] = PSF[:,0:1] / PSF[:,0:1].sum()
        PSF[:,1:2] = PSF[:,1:2] / PSF[:,1:2].sum()  
    
    return dset, PSF, meta


def generate_realdata(path, file, Nz):
    
    dset, PSF, meta = load_real_data(path, file, Nz )
    print(dset.shape)
    device = dset.device
    noise_image = dset
    
    if Nz == 1: 

        physics = dinv.physics.BlurFFT(img_size = (1, 1, meta.nx, meta.ny), filter = PSF, device=device)

        x_0 = physics.A_adjoint(noise_image)           
        x_0 = x_0.sum(0).unsqueeze(0)

        x_center = PSF[12:13].sum()
        index = torch.zeros(25, device = device)
        finger_print = torch.zeros(25, device = device)

        for j in range(0,25):
                finger_print[j] = PSF[j:j+1].sum()
                index[j] = finger_print[j]/x_center.to(device)
                
        back_vec = index * 1e-4

        L = torch.zeros(25)
        for j in range(0,25):
                x_ones = torch.ones_like(noise_image.sum(0).repeat(25,1,1,1))
                norm_H = torch.max(physics(x_ones)[j]) * torch.max(physics.A_adjoint(x_ones)[j])
                norm_y = torch.max(torch.abs(noise_image[j]))
                L[j] = (norm_y / back_vec[j]**2)* norm_H
        L_th = torch.sum(L)
        
    if Nz == 2:
        
        physics = dinv.physics.BlurFFT(img_size = (1,2,meta.nx,meta.ny), filter = PSF, device=device)
        # physics_blurr_forw = dinv.physics.BlurFFT(img_size = (1,2,meta.nx,meta.ny), filter = PSF[:,1].unsqueeze(1), device=device)
        # physics_blurr_out = dinv.physics.BlurFFT(img_size = (1,2,meta.nx,meta.ny), filter = PSF[:,0].unsqueeze(1), device=device)

        x_0 = physics.A_adjoint(noise_image)           
        x_0 = x_0.sum(0).unsqueeze(0)

        x_center = PSF[12:13,0].sum()
        index = torch.zeros(25, device = device)
        finger_print = torch.zeros(25, device = device)
        for j in range(0,25):
                finger_print[j] = PSF[j:j+1].sum()
                index[j] = finger_print[j] / x_center.to(device)

        back_vec = index * 1e-4

        L = torch.zeros(25)
        for j in range(0,25):
                x_ones = torch.ones_like(noise_image.sum(0).repeat(25,1,1,1))
                norm_H = torch.max(physics(x_ones)[j]) * torch.max(physics.A_adjoint(x_ones)[j])
                norm_y = torch.max(torch.abs(noise_image[j]))
                L[j] = (norm_y / back_vec[j]**2)* norm_H
        L_th = torch.sum(L)
    
        
    
    return noise_image, PSF, meta, finger_print, physics, back_vec, L_th, x_0




def info_from_realdata(noise_image, meta, PSF, Nz):

    
    if Nz == 1: 
            
        PSF = PSF[:,1:2,:,:]

        physics = dinv.physics.BlurFFT(img_size = (1, 1, meta.nx, meta.ny), filter = PSF, device=device)

        x_0 = physics.A_adjoint(noise_image)           
        x_0 = x_0.sum(0).unsqueeze(0)

        x_center = PSF[12:13].sum()
        index = torch.zeros(25, device = device)
        finger_print = torch.zeros(25, device = device)

        for j in range(0,25):
                finger_print[j] = PSF[j:j+1].sum()
                index[j] = finger_print[j]/x_center.to(device)
                
        back_vec = index * 1e-4

        L = torch.zeros(25)
        for j in range(0,25):
                x_ones = torch.ones_like(noise_image.sum(0).repeat(25,1,1,1))
                norm_H = torch.max(physics(x_ones)[j]) * torch.max(physics.A_adjoint(x_ones)[j])
                norm_y = torch.max(torch.abs(noise_image[j]))
                L[j] = (norm_y / back_vec[j]**2)* norm_H
        L_th = torch.sum(L)
        
    if Nz == 2:
        
        physics = dinv.physics.BlurFFT(img_size = (1,2,meta.nx,meta.ny), filter = PSF, device=device)
        # physics_blurr_forw = dinv.physics.BlurFFT(img_size = (1,2,meta.nx,meta.ny), filter = PSF[:,1].unsqueeze(1), device=device)
        # physics_blurr_out = dinv.physics.BlurFFT(img_size = (1,2,meta.nx,meta.ny), filter = PSF[:,0].unsqueeze(1), device=device)

        x_0 = physics.A_adjoint(noise_image)           
        x_0 = x_0.sum(0).unsqueeze(0)

        x_center = PSF[12:13,0].sum()
        index = torch.zeros(25, device = device)
        finger_print = torch.zeros(25, device = device)
        for j in range(0,25):
                finger_print[j] = PSF[j:j+1].sum()
                index[j] = finger_print[j] / x_center.to(device)

        back_vec = index * 1e-4

        L = torch.zeros(25)
        for j in range(0,25):
                x_ones = torch.ones_like(noise_image.sum(0).repeat(25,1,1,1))
                norm_H = torch.max(physics(x_ones)[j]) * torch.max(physics.A_adjoint(x_ones)[j])
                norm_y = torch.max(torch.abs(noise_image[j]))
                L[j] = (norm_y / back_vec[j]**2)* norm_H
        L_th = torch.sum(L)
    
        
    
    return finger_print, physics, back_vec, L_th, x_0



