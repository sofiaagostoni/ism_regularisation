
import torch
import deepinv as dinv
import numpy as np
import torch
import torch.nn as nn

# KL computed with the transformation already done
def KL(x, y, back_vec, physics):
    """Calculate the function KL(x; y) as defined in the problem.
    x = Hx + b"""
    z = physics(x) + back_vec.view(25,1,1,1)
    I0 = (y == 0)
    I1 = (y != 0)
    kl = torch.zeros_like(y)
    term1 = torch.sum(z[I0])
    term2 = torch.sum(y[I1] * torch.log(y[I1] /(z[I1] + 1e-14)) + z[I1] - y[I1])
    kl = term1 + term2
    return kl


class KL(nn.Module):
    def __init__(self, back=1e-3):
        super().__init__()
        self.eps = back  # self.eps funge da background/eps

    def _get_eps(self):
        """Metodo di supporto per formattare eps (back) per il broadcasting."""
        if isinstance(self.eps, torch.Tensor) and self.eps.shape == torch.Size([25]):
            return self.eps.view(25, 1, 1, 1)
        return self.eps

    def forward(self, y, x, physics, alpha=1):
        ax = physics.A(x)  # Inserito l'operatore physics mancante
        ax_alpha = ax * alpha
        val = ax_alpha + self._get_eps()
        
        # Correzione matematica: se y=0, il risultato della divergenza è 'val' (cioè ax_alpha + eps).
        # Nel tuo codice originale restituiva solo 'ax_alpha', ignorando l'eps.
        kl = torch.where(y > 0, y * torch.log(y / val) + val - y, val)
        return torch.sum(kl, dim=(1, 2, 3))
    
    def forward_25(self, y, x, physics):
            # 1. Calcola l'operatore fisico UNA sola volta sull'immagine x (batch=1)
            ax = physics.A(x) 
            
            # 2. Il broadcasting espanderà automaticamente 'ax' a 25 immagini 
            # quando lo sommi a self._get_eps() che ha dimensione 25.
            z = ax + self._get_eps()

            kl = torch.where(y > 0, y * torch.log(y / (z + 1e-14)) + z - y, z)
            return torch.sum(kl)
        
        
    def forward_25_3D(self, y, x, physics):
        x = x.repeat(25,1,1,1)
        clean = physics(x) 
        z = clean.sum(1).unsqueeze(1) + self.eps.view(25,1,1,1) 
        I0 = (y == 0)
        I1 = (y != 0)
        kl = torch.zeros_like(y)
        
        term1 = torch.sum(z[I0] )
        term2 = torch.sum((y[I1] * torch.log(y[I1] /(z[I1] + 1e-14)) + z[I1] - y[I1]))
        
        kl = term1 + term2
        return kl.unsqueeze(0)

    def grad_25_3D(self, y, x, physics):
        x = x.repeat(25,1,1,1)
        clean = physics(x) 
        z = clean.sum(1).unsqueeze(1) + self.eps.view(25,1,1,1) 
        temp1 = physics.A_adjoint(1 - y / z)
        return torch.sum(temp1,0).unsqueeze(0)

    def grad(self, y, x, physics, alpha=1):
        ax = physics.A(x)
        val = alpha * ax + self._get_eps()
        # Aggiunto 1e-14 a denominatore per massima stabilità
        return alpha * physics.A_adjoint(1.0 - y / (val + 1e-14))
    
    def grad_25(self, y, x, physics):
        
        x_rep = x.repeat(25, 1, 1, 1)
        
        # Richiamiamo grad con l'ordine corretto degli argomenti
        temp = self.grad(y, x_rep, physics, alpha=1)

        return torch.sum(temp, dim=0).unsqueeze(0)

def KL_metric(y, x):
    """Calculate the function KL(x; y) as defined in the problem.
    x = Hx + b"""
    z = x
    I0 = (y == 0)
    I1 = (y != 0)
    kl = torch.zeros_like(y)
    term1 = torch.sum(z[I0])
    term2 = torch.sum(y[I1] * torch.log(y[I1] /(z[I1] + 1e-14)) + z[I1] - y[I1])
    kl = term1 + term2
    return kl


def grad_KL(x, y, back, physics):
    return physics.A_adjoint(1 - y / (physics(x) + back))


def KL_noise(x, y):
    """Calculate the function KL(x; y) as defined in the problem.
    x = Hx + b"""
    z = x
    I0 = (y == 0)
    I1 = (y != 0)
    kl = torch.zeros_like(y)
    term1 = torch.sum(z[I0])
    term2 = torch.sum(y[I1] * torch.log(y[I1] /(z[I1] + 1e-14)) + z[I1] - y[I1])
    kl = term1 + term2
    return kl

def grad_KL_multifilter(x_batch, y_batch, eps, physics):
    """
    Calcola il gradiente del KL rispetto a x_batch usando physics.
    y_batch: (B, N_filters, C, H, W)
    x_batch: (B, C, H, W)
    physics: oggetto physics con forward e adjoint
    """
    # --- Aggiunge dimensione batch se manca ---
    if y_batch.ndim == 4:  # (N_filters, C, H, W)
        y_batch = y_batch.unsqueeze(0)  # diventa (1, N_filters, C, H, W)
    # Forward
    Ax = physics(x_batch)  # (B, N_filters, C, H, W)
    # KL ratio
    ratio = 1.0 - y_batch / (Ax + eps.view(25,1,1,1))
    # Adjoint
    Astar_ratio = physics.A_adjoint(ratio)
    # Se l'adjoint restituisce 5D, somma sui filtri
    if Astar_ratio.ndim == 5:
        Astar_ratio = Astar_ratio.sum(dim=1)  # (B, C, H, W)
    # Aggiungiamo dim canale se necessario
    if Astar_ratio.ndim == 3:  # (B,H,W)
        Astar_ratio = Astar_ratio.unsqueeze(1)
    return Astar_ratio  # (B,C,H,W)


def grad_KL_noise(x, y, back):
    return (1 - y / (x+ back))


def grad_KL_25(x, y, back, physics):
    
    if back.shape == torch.Size([25]):
        temp = grad_KL(x.repeat(25,1,1,1),y, back.view(25,1,1,1), physics)
    elif back.shape == torch.Size([25,1,256,256]):
        temp = grad_KL(x.repeat(25,1,1,1),y, back, physics)

    return torch.sum(temp,0).unsqueeze(0)

def zero_reg(x, lam):
    return 0

def KL_25(x, y, back, physics):
    """Calculate the function KL(x; y) as defined in the problem.
    x = Hx + b"""
    x = x.repeat(25,1,1,1)
    if back.shape == torch.Size([25]):
        z = physics(x) +  back.view(25,1,1,1)
    elif back.shape == torch.Size([25,1,256,256]):
        z = physics(x) +  back

    I0 = (y == 0)
    I1 = (y != 0)
    kl = torch.zeros_like(y)
    
    term1 = torch.sum(z[I0] )
    term2 = torch.sum((y[I1] * torch.log(y[I1] /(z[I1] + 1e-14)) + z[I1] - y[I1]))
    
    kl = term1 + term2
    return kl.unsqueeze(0)

def KL_25_3D(x, y, back, physics):
    """Calculate the function KL(x; y) as defined in the problem.
    x = Hx + b"""
    x = x.repeat(25,1,1,1)
    clean = physics(x) 
    z = clean.sum(1).unsqueeze(1) + back.view(25,1,1,1) 
    I0 = (y == 0)
    I1 = (y != 0)
    kl = torch.zeros_like(y)
    
    term1 = torch.sum(z[I0] )
    term2 = torch.sum((y[I1] * torch.log(y[I1] /(z[I1] + 1e-14)) + z[I1] - y[I1]))
    
    kl = term1 + term2
    return kl.unsqueeze(0)

def grad_KL_25_3D(x, y, back, physics):
    x = x.repeat(25,1,1,1)
    clean = physics(x) 
    z = clean.sum(1).unsqueeze(1) + back.view(25,1,1,1) 
    temp1 = physics.A_adjoint(1 - y / z)
    return torch.sum(temp1,0).unsqueeze(0)





