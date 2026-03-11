import os
from . import *
import numpy as np
import torch
from tqdm.auto import tqdm
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from skimage.metrics import structural_similarity
import torchmin
import torch
import math
from .projected_gradient import *


def standardize(Y, lam):
    """
    lam: scalare (float o tensor)
    """
    lam = torch.as_tensor(lam, device=Y.device, dtype=Y.dtype)
    Z = (Y - lam) / torch.sqrt(lam)
    return Z


def T_fun(lam):
    """Calcola T(λ) = 1 / (1 - exp(-λ)) in float64"""
    # Cast a float64 per il calcolo interno
    lam64 = lam.to(torch.float64)
    
    # Calcolo con maggiore precisione
    res = 1.0 / (1.0 - torch.exp(-lam64))
    
    # Ritorna in float32
    return res.to(torch.float32)

def V_fun(lam):
    """Calcola V(λ) in float64"""
    lam64 = lam.to(torch.float64)
    exp_neg_lam = torch.exp(-lam64)
    
    # Formula: (1 - (1+λ)exp(-λ)) / (1 - exp(-λ))^2
    num = 1.0 - (1.0 + lam64) * exp_neg_lam +1e-10
    den = (1.0 - exp_neg_lam)**2 + 2e-10
    
    # Nota: Rimosso l'epsilon manuale poiché float64 gestisce meglio i valori piccoli.
    # Se lam è esattamente 0, den sarà 0.
    res = num / den
    
    return res.to(torch.float32)


def compute_truncation_stats_eps(lam, eps):
    """
    Calcola il Valore Atteso e la Varianza per una Poisson troncata Y > eps.
    Lavora in float64 per evitare errori di arrotondamento.
    """
    lam64 = lam.to(torch.float64)
    k = int(math.floor(eps))
    
    # Inizializzazione delle sommatorie
    S0 = torch.zeros_like(lam64)
    S1 = torch.zeros_like(lam64)
    S2 = torch.zeros_like(lam64)
    
    exp_neg_lam = torch.exp(-lam64)
    
    # Calcolo delle somme per i termini scartati (da z=0 a k)
    for z in range(k + 1):
        # P(Y=z) = exp(-lam) * lam^z / z!
        prob_z = exp_neg_lam * (lam64 ** z) / math.factorial(z)
        
        S0 += prob_z                # sum till k of P_Y
        S1 += z * prob_z            # sum till k of zP_Y
        S2 += (z ** 2) * prob_z     # sum till k of z^2 P_Y
        
    # T_eps(lambda) = 1 / (1 - S0)
    # Aggiungo un piccolo epsilon al denominatore per stabilità numerica se S0 -> 1
    T_eps = 1.0 / (1.0 - S0 + 1e-12)
    
    # Valore Atteso (Media)
    E_eps = T_eps * (lam64 - S1)
    
    # Varianza
    # Formula: Var = E[Y^2] - (E[Y])^2
    Var_eps = T_eps * (lam64 * (1.0 + lam64) - S2) - (E_eps ** 2)
    
    # Clamp per evitare varianze negative/nulle dovute ad approssimazioni di macchina
    Var_eps = torch.clamp(Var_eps, min=1e-12)
    
    return E_eps.to(torch.float32), Var_eps.to(torch.float32)


def standardize_unbiased_masked(Y, lam):
    """
    Calcola la versione 'masked' z_+(U) come definita in Sezione 4.5.
    
    Y: tensore delle osservazioni (y)
    lam: tensore della media stimata (λ_hat)
    """
    # 1. Calcolo di T e V
    T = T_fun(lam)
    V = V_fun(lam)
    
    # 2. Standardizzazione secondo la formula (28):
    # z(U) = (y - λ*T(λ)) / sqrt(λ*V(λ))
    Z = (Y - lam * T) / torch.sqrt(lam * V)
    
    # 3. Applicazione del blind masking (I_+):
    # Restituisce z_i se i appartiene a I_+, altrimenti 0.
    # Assumendo I_+ come l'indice delle Y positive (y_i > 0).
    Z_plus = torch.where(Y > 0, Z, torch.zeros_like(Z))
    
    return Z_plus


def standardize_unbiased_masked_eps(Y, lam, eps):
    """
    Calcola la versione 'masked' Z_eps per Y > eps.
    
    Y: tensore delle osservazioni (y)
    lam: tensore della media stimata (λ_hat)
    eps: scalare, soglia di troncamento (es. 0, 1.5, 3)
    """
    # 1. Calcolo di Valore Atteso e Varianza troncati
    E_eps, Var_eps = compute_truncation_stats_eps(lam, eps)
    
    # 2. Standardizzazione: Z_eps = (Y - E_eps) / sqrt(Var_eps)
    Z_eps = (Y - E_eps) / torch.sqrt(Var_eps)
    
    # 3. Applicazione del blind masking:
    # Restituisce Z_eps se Y > eps, altrimenti 0.
    Z_eps_masked = torch.where(Y > eps, Z_eps, torch.zeros_like(Z_eps))
    
    return Z_eps_masked

def whiteness_measure(Z: torch.Tensor) -> torch.Tensor:
    """
    Calcola W(Z) utilizzando la Proposizione 2.1 (dominio delle frequenze).
    Assume condizioni al contorno periodiche.
    
    Z: tensor 3D (m1, m2, m3), reale.
    Ritorna: scalare (Tensor) rappresentante la misura di bianchezza.
    """
    if Z.ndim != 3:
        # Se Z ha una dimensione batch o canale (es. 1, m1, m2, m3), la rimuoviamo
        Z = Z.squeeze()
        if Z.ndim != 3:
            raise ValueError(f"Z deve essere 3D, ma ha shape {Z.shape}")

    # 1. Trasformata di Fourier 3D (non serve padding con condizioni periodiche)
    # Usiamo rfftn (Real FFT) per efficienza se Z è reale
    FZ = torch.fft.fftn(Z)
    
    # 2. Calcolo dei moduli al quadrato |z_tilde|^2
    # Nota: abs() su un complesso restituisce il modulo, eleviamo alla seconda
    mag_sq = torch.abs(FZ)**2
    
    # 3. Numeratore: Somma dei moduli alla quarta power
    numerator = torch.sum(mag_sq**2)
    
    # 4. Denominatore: Quadrato della somma dei moduli al quadrato
    denominator = torch.sum(mag_sq)**2
    
    if denominator == 0:
        raise ValueError(f"Denominatore uguale a zero")
        
    return numerator / denominator


def RWP(mu_values_grid, noise_image, back_vec, parameters, optim = Pgd_Backtracking,
        algorithm="pgd", is_3d=False, is_realdata=False,
        mask_type="masked", eps=1):
    """
    Calcola il Residual Whiteness Principle (RWP) per una griglia di parametri mu (lambda).
    Sfrutta le classi OOP (PGDSolver, ProxSolver) per massima efficienza e pulizia.
    """
    # 1. RECUPERA IL DEVICE DAI DATI IN INGRESSO
    device = noise_image.device 

    M = noise_image.numel() 
    n = len(mu_values_grid)
    
    # 2. ASSICURATI CHE I TENSORI VENGANO CREATI SUL DEVICE CORRETTO
    W_sum = torch.empty(n, device=device)    
    psnr_vecs = torch.empty(n, device=device) if not is_realdata else None 
    ssim_vecs = torch.empty(n, device=device) if not is_realdata else None   
    
    min_distance = float('inf') 
    best_x_w = None
    
    # --- 1. SETUP DEL SOLVER ---
    SolverClass = optim
    solver = SolverClass(parameters, algorithm = algorithm, is_3d=is_3d, is_realdata = is_realdata)

    physics = parameters["physics"]

    # --- 2. PRE-CALCOLO GROUND TRUTH E Z_TRUE (Solo per dati simulati) ---
    wh_true = None
    if not is_realdata:
        ground_truth = parameters["ground_truth"]
        # Uso .view(-1, 1, 1, 1) così si adatta automaticamente a Nz (es. 25 o 2)
        clean_image = physics(ground_truth) + back_vec.view(-1, 1, 1, 1)
        
        clean_image_proc = clean_image.sum(1).unsqueeze(1) if is_3d else clean_image
        
        if mask_type == "masked":
            Z_true = standardize_unbiased_masked(noise_image, clean_image_proc)
        elif mask_type == "masked_eps":
            Z_true = standardize_unbiased_masked_eps(noise_image, clean_image_proc, eps)
        elif mask_type == "whole":
            Z_true = standardize(noise_image, clean_image_proc)
        else:
            raise ValueError(f"Metodo di masking non riconosciuto: {mask_type}")

        wh_true = whiteness_measure(Z_true)

    # --- 3. RICERCA SULLA GRIGLIA MU ---
    for i, mu in enumerate(tqdm(mu_values_grid, desc="Searching mu grid (RWP)")):
        print(f"\n--- Testing mu parameter = {mu} ---")
        
        parameters['lam'] = mu
        
        # Esecuzione del solver elegantissima: tutto è già configurato dentro l'oggetto!
        results = solver.solve(y=noise_image)
        x_result = results['x_result']
      
        # Calcolo e riadattamento di lambda_d
        lambda_d = physics(x_result) + back_vec.view(-1, 1, 1, 1)
        if is_3d:
            lambda_d = lambda_d.sum(1).unsqueeze(1)
        
        # Calcolo di Z per il risultato corrente
        if mask_type == "masked":
            Z = standardize_unbiased_masked(noise_image, lambda_d) 
        elif mask_type == "masked_eps":
            Z = standardize_unbiased_masked_eps(noise_image, lambda_d, eps)
        elif mask_type == "whole":
            Z = standardize(noise_image, lambda_d)
            
        # Metriche Whiteness
        wh = whiteness_measure(Z)
        W_sum[i] = M * wh
        
        # Metriche PSNR/SSIM (solo se non siamo con dati reali)
        if not is_realdata:
            # results['psnr'] contiene l'evoluzione, prendiamo l'ultimo elemento [-1]
            psnr_vecs[i] = results['psnr'][-1].item()
            ssim_vecs[i] = results['ssim'][-1].item()
            
            print(f"PSNR = {psnr_vecs[i]:.2f} | SSIM = {ssim_vecs[i]:.4f}")
        
        # Aggiornamento del minimo
        if W_sum[i] < min_distance:
            best_x_w = x_result
            min_distance = W_sum[i]

        print(f"WP = {W_sum[i]:.4f}")

    return W_sum, psnr_vecs, ssim_vecs, best_x_w, wh_true