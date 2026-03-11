
import torch
import deepinv as dinv
import numpy as np
from torch import nn


# gradient and laplacian

def grad(image):
    """
    Calcola il gradiente (grad_x, grad_y) con differenze finite backward
    e condizioni al bordo periodiche
    
    Restituisce:
    - grad_x: Derivata parziale rispetto a x, stessa forma di image (1, C, n1, n2).
    - grad_y: Derivata parziale rispetto a y, stessa forma di image (1, C, n1, n2).
    - laplacian: Laplaciano dell'immagine, stessa forma di image (1, C, n1, n2).
    """
    # Forward differences (f(x+1,y) - f(x,y))

    grad_x = torch.roll(image, shifts=-1, dims=2) - image  # x-axis
    grad_y = torch.roll(image, shifts=-1, dims=3) - image  # y-axis

    return grad_x, grad_y

def divergen(grad_1, grad_2):
    # Compute divergence using backward finite differences with periodic boundary conditions
    div_1 = grad_1 - torch.roll(grad_1, shifts=1, dims=2)  # Difference along x (axis 2)
    div_2 = grad_2 - torch.roll(grad_2, shifts=1, dims=3)  # Difference along y (axis 3)
    
    div = div_1 + div_2
    return div

def laplacian(image):
    """ 
    compute divergence of the gradient of the image 
    """
    # Calcolo del Laplaciano: somma delle derivate seconde

    grad_x, grad_y = grad(image)  # Calcola il gradiente
    laplac = divergen(grad_x, grad_y)  # Calcola la divergenza del gradiente
    return laplac


# TV reg

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1.0, eps=1e-2):
        super().__init__()
        self.tv_loss_weight = tv_loss_weight
        self.eps = eps
    def forward(self, x):
        # differenze orizzontali e verticali
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        tv = torch.sqrt(dx[:, :, :, :-1]**2 + dy[:, :, :-1, :]**2 + self.eps)
        return self.tv_loss_weight * tv.sum(dim=(1, 2, 3))
    def grad(self, x):
        """
        Calcola il gradiente tramite autograd senza mantenere il grafo.
        """
        x = x.clone().requires_grad_(True)
        loss = self.forward(x)
        grad = torch.autograd.grad(
            outputs=loss.sum(), inputs=x, retain_graph=False
        )[0]
        return grad
tv=TVLoss()



eps_tv = 1e-6



def total_variation_3D(image, eps = eps_tv):
    """
    Applica la regolarizzazione di total variation
    """
    image = image[:,1].unsqueeze(1)
    grad_x, grad_y = grad(image)
    norm_gradient = grad_x**2 + grad_y**2  # Modulo quadrato del gradiente

    tot_var = torch.sum(torch.sqrt(norm_gradient + eps**2 ))

    return tot_var

def total_variation_grad_3D(image, eps = eps_tv):
    
    image = image[:,1].unsqueeze(1)

    grad_x, grad_y = grad(image)
    norm_gradient = grad_x**2 + grad_y**2  # Modulo quadrato del gradiente

    denom = torch.sqrt( norm_gradient + eps**2)

    new_grad_x = grad_x / denom
    new_grad_y = grad_y / denom
    new_grad = - divergen(new_grad_x, new_grad_y)
    tv_grad =  new_grad
    
    return tv_grad

# Tikhonov reg order 1

def tik_0(x, lam ):
    f = lam * 0.5 * torch.norm(x) ** 2
    return f

def grad_tik_0(x, lam):
    grad = lam * x
    return grad



# sobolev reg (tikhonov order 2)

def sobolev(image):
    """
    Applica la regolarizzazione di Sobolev di ordine 1 (Tikhonov) 
    """
    grad_x, grad_y = grad(image)
    norm_gradient = grad_x**2 + grad_y**2  # Modulo quadrato del gradiente
    
    sobolev_energy =  0.5 * torch.sum(norm_gradient)  # Funzione di regolarizzazione
    return sobolev_energy

def sobolev_grad(image):
    """
    Calcola il gradiente della regolarizzazione di Sobolev
    """
    laplacian_image = laplacian(image)
    sobolev_grad = - laplacian_image
    
    return sobolev_grad

def l1_energy(image):
    """
    Regolarizzazione L1 lisciata: ∑ sqrt(|grad u|^2 + eps)
    """
    return torch.norm(image, p=1)
eps_l1=1e-6

def l1_smooth_energy(image):
    """
    Regolarizzazione L1 lisciata: ∑ sqrt(|grad u|^2 + eps)
    """
    return torch.sqrt(image**2 + eps_l1**2).sum()


def l1_smooth_grad(image):
    """
    Gradiente della regolarizzazione L1 lisciata:
    -div( grad u / sqrt(|grad u|^2 + eps) )
    """
    return image / torch.sqrt(image**2 + eps_l1**2)



