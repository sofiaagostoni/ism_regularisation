import torch
import deepinv as dinv
from deepinv.optim.data_fidelity import PoissonLikelihood, L2
from deepinv.physics import Denoising, PoissonNoise

        
import ISM.simulation.PSF_sim as ism

import torch
import deepinv as dinv
from deepinv.physics import Blur, PoissonNoise

class PoissonWithBackground(PoissonNoise):
    def __init__(self, gain, bkg):
        super().__init__(gain=gain)
        self.bkg = bkg

    def forward(self, x, **kwargs):
        return super().forward(x + self.bkg, **kwargs)



def generate_meas_ism(image, Nx, Nz, pxsize, flux, device):
        
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
    PSF, _, _ = ism.SPAD_PSF_2D(grid, exPar, emPar)
        
    psf_center = PSF[12:13] #center
    # fingerprint of center psf
    f_center = torch.sum(psf_center).to(device)
    # background weight parameter
    beta = torch.tensor(1e-4).to(device)

    # compute normalization
    x_center = PSF[12:13].sum()
    index = torch.zeros(25).to(device)
    finger_print = torch.zeros(25).to(device)
    for j in range(0,25):
        finger_print[j] = PSF[j:j+1].sum()
        index[j] = finger_print[j]/x_center
    eta_vec = (beta * finger_print / f_center).squeeze()
    
    n_channels = 1  # 3 for color images, 1 for gray-scale images
    # create stacked physics operators, one for each kernel
    # and precompute the background per kernel (aka detector)
    physics_list = []
    rec_alpha = 1/flux

    for i in range(PSF.shape[0]):
        physics_blurr_i = dinv.physics.BlurFFT(
            img_size=(n_channels, Nx, Nx),
            filter=PSF[i].unsqueeze(0),
            device=device,
        )
        physics_noise_i = Denoising()
        physics_noise_i.noise_model = PoissonWithBackground(gain = rec_alpha, bkg=eta_vec[i])
        physics_list.append(physics_noise_i * physics_blurr_i)
        
    physics = dinv.physics.StackedLinearPhysics(physics_list, device=device)


    like_list_poisson = []
    for i in range(len(physics_list)):
        like_i_poisson = PoissonLikelihood(gain=1.0/flux, bkg=eta_vec[i])
        like_list_poisson.append(like_i_poisson)

    data_fidelity_poisson = dinv.optim.StackedPhysicsDataFidelity(like_list_poisson)
    
    like_list_l2 = []
    for i in range(len(physics_list)):
        like_i_l2 = L2(sigma=1.0)
        like_list_l2.append(like_i_l2)

    data_fidelity_l2 = dinv.optim.StackedPhysicsDataFidelity(like_list_l2)

    # Apply the degradation to the image
    y = physics(image)

    #compute the mean
    avg_y= [sum(y_i)/len(y) for y_i in zip(*y)]
    # compute the sum
    sum_y = [sum(y_i) for y_i in zip(*y)]


    # Lipschitz costant
    L_kl = torch.zeros(25, device=device)
    x_ones = torch.ones_like(image)
    for j in range(25):
        A_j = physics_list[j]
        norm_H = A_j.A(x_ones).max() * A_j.A_adjoint(torch.ones_like(y[j])).max()
        norm_y = y[j].abs().max()
        L_kl[j] = norm_y / eta_vec[j]**2 * norm_H
    L_kl = L_kl.sum()
    
    
    sigma_vec = torch.ones(25, device=device)   # or your per-detector sigmas

    L_l2 = torch.zeros(25, device=device)
    x_ones = torch.ones_like(image)
    for j in range(25):
        A_j = physics_list[j]
        norm_H = A_j.A(x_ones).max() * A_j.A_adjoint(torch.ones_like(y[j])).max()
        L_l2[j] = norm_H / sigma_vec[j]**2
    L_l2 = L_l2.sum()

        
    
    return PSF, y, avg_y, sum_y, finger_print, physics, data_fidelity_poisson, data_fidelity_l2, L_kl, L_l2

def crop_center(img, cropx, cropy):
        y, x = img.shape[2], img.shape[3]
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[:,:,starty : starty + cropy, startx : startx + cropx]
    
class SummedBlurPhysics(dinv.physics.Physics):
    def __init__(self, n_channels, Nx, kernel1, kernel2, noise_model=None, device='cpu'):
        super().__init__()
        # Definiamo i due operatori di blur
        # Assumiamo che x sia un tensore che concatena x1 e x2 lungo la dimensione dei canali o una nuova dimensione
        self.A1 = dinv.physics.BlurFFT(img_size=(n_channels, Nx, Nx),filter=kernel1, device=device)
        self.A2 = dinv.physics.BlurFFT(img_size=(n_channels, Nx, Nx),filter=kernel2, device=device)
        self.noise_model = noise_model

    def A(self, x):
        """
        Input x: si assume diviso in due parti (es. lungo i canali)
        x = [x1, x2]
        """
        # Esempio: x1 e x2 sono divisi lungo la dimensione 1 (canali)
        
        # Applichiamo i blur e sommiamo nel dominio lineare (pre-noise)
        return self.A1.A(x[:,0:1]) + self.A2.A(x[:,1:2])

    def A_adjoint(self, y):
        # L'aggiunto di una somma è il vettore degli aggiunti
        adj1 = self.A1.A_adjoint(y)
        adj2 = self.A2.A_adjoint(y)
        return torch.cat([adj1, adj2], dim=1)    
    
def generate_meas_ism_3D(x, Nx, Nz, pxsizex, flux, device):


    grid = ism.GridParameters()

    grid.N = 5              # number of detector elements in each dimension
    grid.pxsizex = pxsizex      # pixel size of the simulation space (nm)
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
    PSF = PSF.permute(3, 0, 1, 2).to(device)
    PSF[:,0:1] = PSF[:,0:1] / PSF[:,0:1].sum()
    PSF[:,1:2] = PSF[:,1:2] / PSF[:,1:2].sum()

    poisson_noise = PoissonNoise(gain=1/flux).to(device)

    n_channels = 1
    physics_list = []
    for i in range(PSF.shape[0]):
        physics_i = SummedBlurPhysics(n_channels, Nx, PSF[i:i+1,0:1] , PSF[i:i+1,1:2], noise_model=poisson_noise).to(device)
        physics_list.append(physics_i)

    # 1. Crea lo StackedLinearPhysics base (solo l'ottica, senza rumore)
    physics = dinv.physics.StackedLinearPhysics(physics_list, device=device)

    y = physics(x).to(device)

    sum_y = [sum(y_i) for y_i in zip(*y)]
    
    # compute fingerprint and background
    x_center = PSF[12:13,0].sum()
    index = torch.zeros(25, device = device)
    finger_print = torch.zeros(25, device = device)
    for j in range(0,25):
        finger_print[j] = PSF[j:j+1].sum()
        index[j] = finger_print[j] / x_center.to(device)
    eta_vec = index * 1e-4
    
    like_list = []
    for i in range(len(physics_list)):
        like_i = PoissonLikelihood(gain=1.0/flux, bkg=eta_vec[i])
        like_list.append(like_i)

    data_fidelity = dinv.optim.StackedPhysicsDataFidelity(like_list)

    
    
    return PSF, y, sum_y, finger_print, physics, data_fidelity


