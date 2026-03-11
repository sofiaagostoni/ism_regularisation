# %%
import torch
import deepinv as dinv
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS
from opt_functions import * 
from .real_data_load import * 

import torchmin
        
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
import time
from scipy.optimize import least_squares

def generate_meas_ism(image, Nx, Nz, pxsize, flux, s, n_samples = 10, mean_on_noise = False, normalization = False):
    
    device = image.device
    
    grid = ism.GridParameters()

    grid.N = 5              # number of detector elements in each dimension
    grid.pxsizex = pxsize       # pixel size of the simulation space (nm)
    grid.pxdim = 50e3       # detector element size in real space (nm)
    grid.pxpitch = 75e3     # detector element pitch in real space (nm)
    grid.M = 500            # total magnification of the optical system (e.g. 100x objective followed by 5x telescope)
    grid.Nz = 1
    grid.pxsizez = 700
    exPar = ism.simSettings()
    exPar.wl = 640 # excitation wavelength (nm)
    exPar.mask_sampl = 31
    emPar = exPar.copy()
    emPar.wl = 660 # emission wavelength (nm)
    z_shift = 0 #nm
    
    # create a 2D PSF for 25 detectors 
    PSF, detPSF, exPSF = ism.SPAD_PSF_2D(grid, exPar, emPar)
    # PSF_crop = crop_center(PSF, 100, 100)
    
    ## DEFINE OPERATORS------------------------------
    physics_blurr = dinv.physics.BlurFFT(img_size = (1, 1, Nx, Nx), filter = PSF, device=device)
    physics_noise = Denoising()
    rec_alpha = 1.0
    physics_noise.noise_model = PoissonNoise(gain = rec_alpha)
    
    ## COMPUTE FINGERPRINT

    x_center = PSF[12:13].sum()
    index = torch.zeros(25, device = device)
    finger_print = torch.zeros(25, device = device)
    for j in range(0,25):
        finger_print[j] = PSF[j:j+1].sum()
        index[j] = finger_print[j]/x_center.to(device)
    back_vec = index * 1e-3

    ground_truth = flux * image.to(device)

    clean_image = physics_blurr(ground_truth.repeat(25,1,1,1))
    clean_image = clean_image + back_vec.view(25,1,1,1)        # add background
    
    if mean_on_noise == True:
        sum_mean = torch.zeros_like(clean_image)
        for n in range(n_samples):
            noisey = physics_noise(clean_image)
            sum_mean += noisey
        noise_image = sum_mean/n_samples
        
    else:
        noise_image = physics_noise(clean_image)
        
    
    if normalization == True:

        # # Normalization of y_i

        for i in range(25):
            sum_y_i = torch.sum(noise_image[i])
            print(f"before normalization {noise_image[i].max()}")
            noise_image[i] = (noise_image[i] / sum_y_i) * finger_print[i] * s
            print(f"after normalization {noise_image[i].max()}")
            
        # for i in range(25):
        #     max_y_i = torch.max(noise_image[i])
        #     print(f"before normalization {noise_image[i].max()}")
        #     noise_image[i] = (noise_image[i] / max_y_i) * finger_print[i]
        #     print(f"after normalization {noise_image[i].max()}")

        # Initial vector
        x_0 = physics_blurr.A_adjoint(noise_image)           
        x_0 = x_0.sum(0).unsqueeze(0)
        x_0 = (x_0 / x_0.sum()) * s

        # Lipschitz costant
        L = torch.zeros(25)
        for j in range(0,25):
            x_ones = torch.ones_like(ground_truth.repeat(25,1,1,1))
            norm_H = torch.max(physics_blurr(x_ones)[j]) * torch.max(physics_blurr.A_adjoint(x_ones)[j])
            norm_y = torch.max(torch.abs(noise_image[j]))
            L[j] = (norm_y / back_vec[j]**2)* norm_H
        L_th = torch.sum(L)
        
    else:
            
        # Initial vector
        x_0 = physics_blurr.A_adjoint(noise_image)           
        x_0 = x_0.sum(0).unsqueeze(0)

        # Lipschitz costant
        L = torch.zeros(25)
        for j in range(0,25):
            x_ones = torch.ones_like(ground_truth.repeat(25,1,1,1))
            norm_H = torch.max(physics_blurr(x_ones)[j]) * torch.max(physics_blurr.A_adjoint(x_ones)[j])
            norm_y = torch.max(torch.abs(noise_image[j]))
            L[j] = (norm_y / back_vec[j]**2)* norm_H
        L_th = torch.sum(L)
        
    
    return PSF, ground_truth, clean_image, noise_image, finger_print, physics_blurr, back_vec, L_th, x_0


   
   
def generate_meas_ism_3D(image, Nx, Nz, pxsize, flux):


    grid = ism.GridParameters()

    grid.N = 5              # number of detector elements in each dimension
    grid.pxsizex = pxsize       # pixel size of the simulation space (nm)
    grid.pxdim = 50e3       # detector element size in real space (nm)
    grid.pxpitch = 75e3     # detector element pitch in real space (nm)
    grid.M = 500            # total magnification of the optical system (e.g. 100x objective followed by 5x telescope)
    grid.Nz = Nz
    grid.pxsizez = 700
    exPar = ism.simSettings()
    exPar.wl = 640 # excitation wavelength (nm)
    exPar.mask_sampl = 31
    emPar = exPar.copy()
    emPar.wl = 660 # emission wavelength (nm)
    z_shift = 0 #nm

    # create a 2D PSF for 25 detectors 
    PSF, detPSF, exPSF = ism.SPAD_PSF_3D(grid, exPar, emPar)
    
  
    # PSF = PSF.unsqueeze(0)
    PSF = PSF.permute(3, 0, 1, 2)
    PSF[:,0:1] = PSF[:,0:1] / PSF[:,0:1].sum()
    PSF[:,1:2] = PSF[:,1:2] / PSF[:,1:2].sum()

    ground_truth = flux * image.to(device)
    ground_truth = ground_truth.unsqueeze(0)

    physics_blurr = dinv.physics.BlurFFT(img_size = (1,2,Nx,Nx), filter = PSF, device=device)
    physics_blurr_forw = dinv.physics.BlurFFT(img_size = (1,2,Nx,Nx), filter = PSF[:,1].unsqueeze(1), device=device)
    physics_blurr_out = dinv.physics.BlurFFT(img_size = (1,2,Nx,Nx), filter = PSF[:,0].unsqueeze(1), device=device)

    physics_noise = Denoising()
    rec_alpha = 1.0
    physics_noise.noise_model = PoissonNoise(gain = rec_alpha)

    # compute fingerprint and background
    x_center = PSF[12:13,0].sum()
    index = torch.zeros(25, device = device)
    finger_print = torch.zeros(25, device = device)
    for j in range(0,25):
        finger_print[j] = PSF[j:j+1].sum()
        index[j] = finger_print[j] / x_center.to(device)
    back_vec = index * 1e-4
    
    clean_image = physics_blurr(ground_truth)
    clean_image_out = clean_image[:,0:1]
    clean_image_in = clean_image[:,1:2]

    clean_image = clean_image.sum(1).unsqueeze(1) + back_vec.view(25,1,1,1) 
    noise_image = physics_noise(clean_image)

    # Initial vector
    x_0 = physics_blurr.A_adjoint(noise_image)         
    x_0 = x_0.sum(0).unsqueeze(0)

    # Lipschitz costant
    L = torch.zeros(25)
    for j in range(0,25):
        x_ones = torch.ones_like(ground_truth.repeat(25,1,1,1))
        norm_H = torch.max(physics_blurr(x_ones)[j]) * torch.max(physics_blurr.A_adjoint(x_ones)[j])
        norm_y = torch.max(torch.abs(noise_image[j]))
        L[j] = (norm_y / back_vec[j]**2)* norm_H
    L_th = torch.sum(L)
    
    return PSF, ground_truth, clean_image, clean_image_out, clean_image_in, noise_image, finger_print, physics_blurr_forw, physics_blurr_out, physics_blurr, physics_noise, back_vec, L_th, x_0



def prepare_ism_data(is_real=False, real_name='convallaria', load_path=None, 
                     phantom_type='tubulin', Nx=256, Ny=256, Nz=2, pxsize=40, flux=10, 
                     device='cpu', show_plots=True):
    """
    Gestisce il caricamento o la generazione dei dati ISM e delle misurazioni.
    Ritorna un dizionario con tutti gli elementi essenziali per il solver.
    Il 3D è gestito automaticamente dal parametro Nz (Nz > 1 implica 3D).
    """
    
    # grid = ism.GridParameters()

    # grid.N = 5              # number of detector elements in each dimension
    # grid.pxsizex = pxsize      # pixel size of the simulation space (nm)
    # grid.pxdim = 50e3       # detector element size in real space (nm)
    # grid.pxpitch = 75e3     # detector element pitch in real space (nm)
    # grid.M = 500            # total magnification of the optical system (e.g. 100x objective followed by 5x telescope)
    # grid.Nz = Nz
    # grid.pxsizez = 700
    # exPar = ism.simSettings()
    # exPar.wl = 640 # excitation wavelength (nm)
    # exPar.mask_sampl = 31
    # emPar = exPar.copy()
    # emPar.wl = 660 # emission wavelength (nm)
    # z_shift = 0 #nm
    # 1. Determiniamo la dimensionalità in un unico punto
    is_3d = (Nz > 1)
    
    # Print dinamico e pulito
    mode_str = "3D" if is_3d else "2D"
    data_str = "Real" if is_real else "Simulated"
    print(f"--- Preparing {data_str} Data ({mode_str}) ---")
    
    # Inizializziamo a None le variabili che non esistono per i dati reali
    ground_truth = None
    clean_final = None

    # ==========================================
    # CASO A: DATI REALI
    # ==========================================
    if is_real:
        data_path = f"Data/Real_data/{real_name}_data.pth"
        print(f"Loading real data from: {data_path}")
        
        data = torch.load(data_path, weights_only=False)
        noise_image = data['measurment'].to(device)
        PSF = data['PSF'].to(device) 
        meta = data['metadati']
        exwl = data['exwl']
        emwl = data['emwl']
        
        exPar = sim.simSettings()
        exPar.wl = 488  #488 per argolight and convallaria, 640 for purkinje
        exPar.mask_sampl = 50

        emPar = exPar.copy()
        emPar.wl = emwl #520 per argolight, 660 for purkinje, 510 for convallaria, 515 for tubuline

        Nx = meta.nx
        Ny = meta.ny
        dx = meta.dx
        dt = meta.pxdwelltime

        grid = sim.GridParameters()
        pxsizex = meta.dx*1e3
        grid.pxsizex = pxsizex
        # grid.Nz = 1        
        
        # info_from_realdata adatterà la fisica in 2D o 3D in base a Nz
        finger_print, physics, back_vec, L_th, x_0 = info_from_realdata(noise_image, meta, PSF, Nz=Nz)
        print(f'Cost L {L_th}')
        
        if is_3d:
            PSF = PSF
        else:
            PSF = PSF[:,1:2,:,:]        

        
        if show_plots:
            clabel = meta.pxdwelltime
            gr.ShowImg(noise_image.sum(0).cpu(), meta.dx, clabel=clabel)
            fig2 = gr.ShowDataset(noise_image.cpu())
            
            if is_3d:
                fig1 = gr.ShowDataset(PSF[:,0:1].cpu(), normalize = False)
                fig2 = gr.ShowDataset(PSF[:,1:2].cpu(), normalize = False)
                
            else:   
                fig = gr.ShowDataset(PSF.cpu(), normalize = False)


    # ==========================================
    # CASO B: DATI SIMULATI
    # ==========================================
    else:
        # --- B1. CARICAMENTO O GENERAZIONE PHANTOM ---
        if load_path is not None:
            print(f"Loading phantom from: {load_path}")
            if is_3d:
                [image1, image2] = torch.load(load_path, map_location=device)
                image = torch.stack([image1, image2]).squeeze()
            else:
                image = torch.load(load_path, map_location=device)
        else:
            print(f"Generating phantom type: {phantom_type}")
            if is_3d:
                image1 = generate_phantom(phantom_type, Nx, Ny, Nz, pxsize).to(device)
                image2 = generate_phantom(phantom_type, Nx, Ny, Nz, pxsize).to(device)
                image = torch.stack([image1, image2]).squeeze()
            else:
                image = generate_phantom(phantom_type, Nx, Ny, Nz, pxsize).to(device)
                
        image = image.to(device)

        # --- B2. GENERAZIONE DELLE MISURAZIONI (FISICA) ---
        if is_3d:
            PSF, ground_truth, clean_image_sum, clean_out, clean_in, noise_image, \
            finger_print, physics_forw, physics_out, physics, physics_noise, back_vec, \
            L_th, x_0 = generate_meas_ism_3D(image, Nx, Nz, pxsize, flux)
            
            clean_final = clean_image_sum
            
            if show_plots:
                import matplotlib.pyplot as plt
                if load_path is None:
                    # assumo tu abbia una funzione plot definita altrove
                    plot([image1, image2], cmap='hot')
                gr.ShowDataset(PSF[:, 0:1].cpu(), normalize=True)
                gr.ShowDataset(PSF[:, 1:2].cpu(), normalize=True)
                gr.ShowImg(ground_truth[:, 0].unsqueeze(1).to("cpu"), pxsize*1e-3)
                gr.ShowImg(ground_truth[:, 1].unsqueeze(1).to("cpu"), pxsize*1e-3)
                gr.ShowDataset(clean_out.cpu(), normalize=True)
                gr.ShowDataset(clean_in.cpu(), normalize=True)
                gr.ShowDataset(noise_image.cpu(), normalize=True)
        
        else: # 2D
            PSF, ground_truth, clean_image, noise_image, finger_print, physics, \
            back_vec, L_th, x_0 = generate_meas_ism(image, Nx, Nz, pxsize, flux, 3e3, n_samples=10)
            
            clean_final = clean_image
            
            if show_plots:
                gr.ShowDataset(PSF.cpu(), normalize=True)
                gr.ShowImg(ground_truth.to("cpu"), pxsize*1e-3)
                gr.ShowDataset(noise_image.cpu(), normalize=True)
                gr.ShowImg(noise_image.sum(0).to("cpu"), pxsize*1e-3)  
                print(f"Max of noise image = {noise_image.max()}")

    # ==========================================
    # 3. CREAZIONE DELL'OUTPUT STANDARD
    # ==========================================
    dataset = {
        "noise_image": noise_image,
        "ground_truth": ground_truth,  # Sarà None se is_real=True
        "clean_image": clean_final,    # Sarà None se is_real=True
        "physics": physics,
        "back_vec": back_vec,
        "fingerprint": finger_print,
        "x_init": x_0,
        "L_th": L_th,
        "PSF": PSF,
        "meta": meta if is_real else None
    }
    
    return dataset
