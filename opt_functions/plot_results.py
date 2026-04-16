import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS
from microssim import MicroSSIM, micro_structural_similarity
from opt_functions import * 
from opt_functions.Solver_functions.white_opt_princ import *
from .Data_manager.generate_measurments import *

import torchmin
from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity
        
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
import time
from scipy.optimize import least_squares
from matplotlib.ticker import ScalarFormatter
import torchvision.utils as vutils



def plot_met(
    functional=None,
    diff_functional=None,
    stop_criterion=None,
    psnr=None,
    ssim=None,
    label="proxgd"
):

    plots = []

    if functional is not None:
        plots.append((functional, "KL Convergence",
                      r"$\mathrm{KL}(A x_k,\ y) + \mu R(x_k)$", "semilogy"))

    if diff_functional is not None:
        plots.append((diff_functional, "KL to Ground Truth",
                      r"$\mathrm{KL}(x_k,\ x_{\mathrm{GT}})$", "semilogy"))

    if stop_criterion is not None:
        plots.append((stop_criterion, "Stopping Criterion",
                      r"$\frac{\|x_k - x_{k+1}\|}{\|x_k\|}$", "semilogy"))

    if psnr is not None:
        plots.append((psnr, "Peak Signal-to-Noise Ratio (PSNR)",
                      r"$\mathrm{PSNR}(x_k,\ x_{\mathrm{GT}})$", "plot"))

    if ssim is not None:
        plots.append((ssim, "Structural Similarity Index (SSIM)",
                      r"$\mathrm{SSIM}(x_k,\ x_{\mathrm{GT}})$", "plot"))

    if len(plots) == 0:
        raise ValueError("Devi fornire almeno una metrica.")

    # Layout
    n_plots = len(plots)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(10 * n_cols, 5 * n_rows))

    axes = np.atleast_1d(axes).flatten()

    for ax, (data, title, ylabel, scale) in zip(axes, plots):
        if scale == "semilogy":
            ax.semilogy(data, label=label)
        else:
            ax.plot(data, label=label)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    # Disattiva subplot inutilizzati
    for ax in axes[len(plots):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()



def plot_results(results, dataset, IS_REAL, IS_3D, pxsize, x0_sec, y0_sec):
    
    Nz = 2 if IS_3D else 1
    
    grid = ism.GridParameters()

    grid.N = 5              # number of detector elements in each dimension
    grid.pxsizex = pxsize      # pixel size of the simulation space (nm)
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
    
    x_result = results['x_result']
    funct = results['funct']
    funct_metric = results['diff_fid']
    iter_err = results['iter_err']
    psnr_vec = results['psnr']
    ssim_vec = results['ssim']

    x_result = x_result.cpu()
    noise_image = dataset["noise_image"].cpu()
    




    if IS_3D and not IS_REAL:
            ground_truth = dataset["ground_truth"].cpu()

            plot([noise_image.sum(0), x_result[:,1:2], x_result[:,0:1] ], cmap = 'hot')

            gr.ShowImg(x_result[:,1:2].to("cpu"), grid.pxsizex*1e-3)
            gr.ShowImg(x_result[:,0:1].to("cpu"), grid.pxsizex*1e-3)

            gr.ShowImg(dataset["ground_truth"][:,1:2].to("cpu"), grid.pxsizex*1e-3)
            gr.ShowImg(dataset["ground_truth"][:,0:1].to("cpu"), grid.pxsizex*1e-3)
            gr.ShowImg(noise_image.sum(0).to("cpu"), grid.pxsizex*1e-3)
            gr.ShowImg(noise_image[12:13].to("cpu"), grid.pxsizex*1e-3)

            plot_met(funct.cpu(), funct_metric.cpu(), iter_err.cpu(), psnr_vec.cpu(), ssim_vec.cpu())

            print(f"PSNR {psnr(ground_truth[:,1:2]/ground_truth[:,1:2].max(), x_result[:,1:2] / x_result[:,1:2].max()).item()}")
            print(f"SSIM {ssim(ground_truth[:,1:2]/ground_truth[:,1:2].max(), x_result[:,1:2] / x_result[:,1:2].max()).item()}")
            print(f"MICROSSIM {micro_structural_similarity(ground_truth[:,1:2].squeeze().detach().cpu().numpy().astype(np.float32), x_result[:,1:2].squeeze().detach().cpu().numpy().astype(np.float32))}")
            
            vutils.save_image(x_result[:,1:2], 'percorso/della/cartella/immagine.png')


    elif not IS_3D and not IS_REAL:
            ground_truth = dataset["ground_truth"].cpu()

            plot([noise_image.sum(0), x_result], cmap = 'hot')

            gr.ShowImg(x_result.to("cpu"), grid.pxsizex*1e-3)

            gr.ShowImg(noise_image.sum(0).to("cpu"), grid.pxsizex*1e-3)
            gr.ShowImg(noise_image[12:13].to("cpu"), grid.pxsizex*1e-3)


            plot_met(funct.cpu(), funct_metric.cpu(), iter_err.cpu(), psnr_vec.cpu(), ssim_vec.cpu())

            print(f"PSNR {psnr(ground_truth/ground_truth.max(), x_result / x_result.max()).item()}")
            print(f"SSIM {ssim(ground_truth/ground_truth.max(), x_result / x_result.max()).item()}")
            print(f"MICROSSIM {micro_structural_similarity(ground_truth.squeeze().detach().cpu().numpy().astype(np.float32), x_result.squeeze().detach().cpu().numpy().astype(np.float32))}")
            
    if IS_REAL and IS_3D:
        
            plot_met(funct.cpu(), funct_metric, iter_err.cpu())

            
            meta = dataset['meta']
            clabel = meta.pxdwelltime
            
            # Assicurati di avere importato anche matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 2, sharex='row', sharey='row', figsize=(10,10))

            x0 = x0_sec
            y0 = y0_sec
            dx = 180
            dy = 180

            # Creiamo DUE rettangoli identici, uno per ogni subplot
            rect1 = patches.Rectangle((y0, x0), dy, dx, linewidth=1, edgecolor='w', facecolor='none')
            rect2 = patches.Rectangle((y0, x0), dy, dx, linewidth=1, edgecolor='w', facecolor='none')

            # --- Prima Colonna (Destra: [0,1] e [1,1]) ---
            gr.ShowImg(x_result[:,1:2,:, :], meta.dx, clabel, fig=fig, ax=ax[0,1])
            gr.ShowImg(x_result[:,1:2,x0:x0+dx, y0:y0+dx], meta.dx, clabel, fig=fig, ax=ax[1,1])
            ax[0,1].add_patch(rect1) # Aggiungiamo il primo rettangolo
            ax[0,1].set_title("Reconstruction")

            # --- Seconda Colonna (Sinistra: [0,0] e [1,0]) ---
            gr.ShowImg(noise_image[:,:,:, :].sum(0), meta.dx, clabel, fig=fig, ax=ax[0,0])
            gr.ShowImg(noise_image[:,:,x0:x0+dx, y0:y0+dx].sum(0), meta.dx, clabel, fig=fig, ax=ax[1,0])
            ax[0,0].add_patch(rect2) # Aggiungiamo il secondo rettangolo
            ax[0,0].set_title("Noise image sum")

            fig.tight_layout()
            
    elif IS_REAL and not IS_3D:
        
            plot_met(funct.cpu(), funct_metric, iter_err.cpu())
            
            meta = dataset['meta']
            clabel = meta.pxdwelltime

            fig, ax = plt.subplots(2,2, sharex = 'row', sharey = 'row', figsize = (10,10))

            x0 = x0_sec
            y0 = y0_sec
            dx = 200
            dy = 200
            # Create a Rectangle patch
            rect = patches.Rectangle((y0, x0), dy, dx, linewidth=1, edgecolor='w', facecolor='none')

            gr.ShowImg(x_result[:,:,:, :], meta.dx, clabel, fig = fig, ax = ax[0,1])
            gr.ShowImg(x_result[:,:,x0:x0+dx, y0:y0+dx], meta.dx, clabel, fig = fig, ax = ax[1,1])
            ax[0,1].add_patch(rect)
            ax[0,1].set_title("Reconstruction")


            gr.ShowImg(noise_image[12:13,:,:, :], meta.dx, clabel, fig = fig, ax = ax[0,0])
            gr.ShowImg(noise_image[12:13,:,x0:x0+dx, y0:y0+dx], meta.dx, clabel, fig = fig, ax = ax[1,0])
            ax[0,0].set_title("Noise image center")
            gr.ShowImg(noise_image.sum(0), meta.dx, clabel, fig = fig, ax = ax[0,0])
            gr.ShowImg(noise_image[:,:,x0:x0+dx, y0:y0+dx].sum(0), meta.dx, clabel, fig = fig, ax = ax[1,0])
            ax[0,0].set_title("Noise image center")
            # ax[1,0].add_patch(rect)

            fig.tight_layout()
            
            
            


def plot_wp_results(mu_values_grid, results, dataset, pxsize, is_real=False, is_3d = True, title="Metrica", layout="twin", x0 = 100, y0 = 100):
    """
    Plotta i risultati dell'ottimizzazione del parametro mu.
    
    Parametri:
    - mu_values_grid: tensore con i valori di mu
    - W_sum: tensore con i valori della loss/WP
    - psnr_vecs: tensore con i valori di PSNR (ignorato se is_real=True)
    - ssim_vecs: tensore con i valori di SSIM (ignorato se is_real=True)
    - is_real: bool, se True non plotta PSNR e SSIM
    - title: stringa per il titolo del grafico (es. "TV", "SOB")
    - layout: "twin" (grafico singolo con doppio asse Y) o "stacked" (3 subplot separati)
    """
    
    W_sum = results["W_sum"]
    psnr_vecs = results["psnr_vecs"]  
    ssim_vecs = results["ssim_vecs"]
    results_best = results["results_best"]
    x_best = results_best["x_result"]
    # Portiamo tutto su CPU per matplotlib
    mu_vals = mu_values_grid.detach().cpu()
    W = torch.abs(W_sum).detach().cpu()
    
    # Calcolo ottimo per WP
    min_idx = torch.argmin(W)
    mu_best_wp = mu_vals[min_idx]
    
    print(f"--- Risultati per {title} ---")
    print(f"Mu ottimale (min WP): {mu_best_wp.item():.6e} con WP = {W[min_idx].item():.6e}")
    
    best_mu_needle = find_knee_point(mu_vals.cpu(),W_sum.cpu())



    # Impostazioni notazione scientifica asse Y
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    if is_real:
        # Caso dati reali: plottiamo SOLO W_sum
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.semilogx(mu_vals, W, label="RWP", color="tab:blue")
        ax1.semilogx(mu_best_wp, W[min_idx], 'go', label = "Minimum")
        
        ax1.yaxis.set_major_formatter(formatter)
        ax1.set_xlabel("$\mu$")
        ax1.set_ylabel("$W(\mu)$", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.axvline(mu_best_wp.item(), color="green", linestyle="--", alpha=0.7)
        ax1.axvline(best_mu_needle.item(), color="red", linestyle="--", alpha=0.7)
        # ax1.axvline(best_mu_knee.item(), color="orange", linestyle="--", alpha=0.7)
        # ax1.axvline(best_mu_two.item(), color="black", linestyle="--", alpha=0.7)
        # ax1.axvline(best_mu_min.item(), color="orange", linestyle="--", alpha=0.7)
        
        
        # 1. Coordinate degli estremi per la corda
        x_chord = [mu_vals[1], mu_vals[-1]]
        y_chord = [W[1], W[-1]]

        # 2. Coordinate del punto di Knee (assumendo tu abbia calcolato best_mu_knee)
        # Troviamo l'indice del knee point nei dati originali per avere la coordinata Y corretta
        idx_knee = np.argmin(np.abs(mu_vals - best_mu_needle))
        y_knee = W[idx_knee]

        # --- DISEGNO DELLA CORDA ---
        ax1.plot(x_chord, y_chord, color='gray', linestyle='--', linewidth=1.5, label="Chord")

        # --- DISEGNO DEL TRIANGOLO ---
        # Creiamo un poligono che unisce: Inizio, Fine e Knee Point
        triangle_x = [mu_vals[1], mu_vals[-1], best_mu_needle.item()]
        triangle_y = [W[1], W[-1], y_knee]
        
        # Usiamo fill per colorare l'area del triangolo (opzionale)
        ax1.fill(triangle_x, triangle_y, color='orange', alpha=0.2, label="Knee Area")
        
        # Disegniamo i bordi del triangolo per chiarezza
        ax1.plot(triangle_x + [triangle_x[0]], triangle_y + [triangle_y[0]], 
                 color='darkorange', linestyle='-', linewidth=1)

        # Segnamo il Knee Point con un marker specifico
        ax1.plot(best_mu_needle, y_knee, 'ro', markersize=8, label="Knee Point")

        
        ax1.legend(loc='best')
        plt.title(f"{title} Real Data")
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        
        
        meta = dataset['meta']
        noise_image = dataset['noise_image'].cpu()
        x_best = x_best.cpu()
        clabel = meta.pxdwelltime

        
    else:
        # Caso dati simulati: abbiamo PSNR e SSIM
        psnr = psnr_vecs.detach().cpu()
        ssim = ssim_vecs.detach().cpu()
        max_idx_psnr = torch.argmax(psnr)
        max_idx_ssim = torch.argmax(ssim)
        
        print(f"Mu ottimale (max PSNR): {mu_vals[max_idx_psnr].item():.6e} con PSNR = {psnr[max_idx_psnr].item():.4f} dB")
        print(f"Mu ottimale (max SSIM): {mu_vals[max_idx_ssim].item():.6e} con SSIM = {ssim[max_idx_ssim].item():.4f}\n")

        if layout == "twin":
            # Layout con doppio asse Y (WP a sinistra, PSNR a destra)
            fig, ax1 = plt.subplots(figsize=(8, 5))

            ax1.semilogx(mu_vals, W, label="RWP", color="tab:blue")
            ax1.semilogx(mu_best_wp, W[min_idx], 'go')
            ax1.yaxis.set_major_formatter(formatter)
            ax1.set_xlabel("$\mu$")
            ax1.set_ylabel("$W(\mu)$", color="tab:blue")
            ax1.tick_params(axis='y', labelcolor="tab:blue")
            ax1.axvline(mu_best_wp.item(), color="green", linestyle="--", alpha=0.7)
            # ax1.axvline(best_mu_needle.item(), color="orange", linestyle="--", alpha=0.7)
            # ax1.axvline(best_mu_knee.item(), color="orange", linestyle="--", alpha=0.7)
            # ax1.axvline(best_mu_two.item(), color="black", linestyle="--", alpha=0.7)
            # ax1.axvline(best_mu_min.item(), color="brown", linestyle="--", alpha=0.7)




            ax2 = ax1.twinx()
            ax2.semilogx(mu_vals, psnr, label="PSNR", color="tab:orange")
            ax2.semilogx(mu_vals[max_idx_psnr], psnr[max_idx_psnr], 'ro')
            ax2.set_ylabel("PSNR (dB)", color="tab:orange")
            ax2.tick_params(axis='y', labelcolor="tab:orange")
            ax2.axvline(mu_vals[max_idx_psnr].item(), color="red", linestyle="--", alpha=0.7)

            # Combina le legende
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

            plt.title(title)
            plt.tight_layout()
            plt.grid(True)
            plt.show()

        elif layout == "stacked":
            # Layout con 3 subplot separati verticalmente
            fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

            # RWP
            axs[0].semilogx(mu_vals, W, color="tab:blue")
            axs[0].semilogx(mu_best_wp, W[min_idx], 'go')
            axs[0].axvline(mu_best_wp.item(), color="green", linestyle="--")
            # axs[0].axvline(best_mu_needle.item(), color="orange", linestyle="--", alpha=0.7)
            axs[0].yaxis.set_major_formatter(formatter)
            axs[0].set_ylabel("WP")
            axs[0].grid(True)

            # PSNR
            axs[1].semilogx(mu_vals, psnr, color="tab:orange")
            axs[1].semilogx(mu_vals[max_idx_psnr], psnr[max_idx_psnr], 'ro')
            axs[1].axvline(mu_vals[max_idx_psnr].item(), color="red", linestyle="--")
            # axs[1].axvline(best_mu_needle.item(), color="orange", linestyle="--", alpha=0.7)
            axs[1].set_ylabel("PSNR (dB)")
            axs[1].grid(True)

            # SSIM
            axs[2].semilogx(mu_vals, ssim, color="tab:green")
            axs[2].semilogx(mu_vals[max_idx_ssim], ssim[max_idx_ssim], 'go')
            axs[2].axvline(mu_vals[max_idx_ssim].item(), color="green", linestyle="--")
            # axs[2].axvline(best_mu_needle.item(), color="orange", linestyle="--", alpha=0.7)

            axs[2].set_ylabel("SSIM")
            axs[2].set_xlabel("$\mu$")
            axs[2].grid(True)

            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
                       
    plot_results(results_best, dataset, is_real, is_3d, pxsize, x0_sec = x0, y0_sec = y0)
    
    
    
def plot_wp_optim_results(mu_values_grid, results, dataset, pxsize, is_real=False, is_3d = True, title="Metrica", layout="twin"):
    """
    Plotta i risultati dell'ottimizzazione del parametro mu.
    
    Parametri:
    - mu_values_grid: tensore con i valori di mu
    - W_sum: tensore con i valori della loss/WP
    - psnr_vecs: tensore con i valori di PSNR (ignorato se is_real=True)
    - ssim_vecs: tensore con i valori di SSIM (ignorato se is_real=True)
    - is_real: bool, se True non plotta PSNR e SSIM
    - title: stringa per il titolo del grafico (es. "TV", "SOB")
    - layout: "twin" (grafico singolo con doppio asse Y) o "stacked" (3 subplot separati)
    """
    
    W_sum = results["W_sum"]

    results_best = results["results_best"]
    x_best = results_best["x_result"]
    # Portiamo tutto su CPU per matplotlib
    mu_vals = mu_values_grid.detach().cpu()
    W = torch.abs(W_sum).detach().cpu()
    
    # Calcolo ottimo per WP
    min_idx = torch.argmin(W)
    mu_best_wp = mu_vals[min_idx]
    
    print(f"--- Risultati per {title} ---")
    print(f"Mu ottimale (min WP): {mu_best_wp.item():.6e} con WP = {W[min_idx].item():.6e}")

    # Impostazioni notazione scientifica asse Y
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    if is_real:
        # Caso dati reali: plottiamo SOLO W_sum
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(mu_vals, W, label="RWP", color="tab:blue")
        ax1.plot(mu_best_wp, W[min_idx], 'go')
        
        ax1.yaxis.set_major_formatter(formatter)
        ax1.set_xlabel("$\mu$")
        ax1.set_ylabel("$W(\mu)$", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.axvline(mu_best_wp.item(), color="green", linestyle="--", alpha=0.7)
        
        ax1.legend(loc='best')
        plt.title(f"{title} (Dati Reali)")
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        
        
        meta = dataset['meta']
        noise_image = dataset['noise_image'].cpu()
        x_best = x_best.cpu()
        clabel = meta.pxdwelltime

        
    else:
        # Caso dati simulati: abbiamo PSNR e SSIM
        psnr = psnr_vecs.detach().cpu()
        ssim = ssim_vecs.detach().cpu()
        max_idx_psnr = torch.argmax(psnr)
        max_idx_ssim = torch.argmax(ssim)
        
        print(f"Mu ottimale (max PSNR): {mu_vals[max_idx_psnr].item():.6e} con PSNR = {psnr[max_idx_psnr].item():.4f} dB")
        print(f"Mu ottimale (max SSIM): {mu_vals[max_idx_ssim].item():.6e} con SSIM = {ssim[max_idx_ssim].item():.4f}\n")

        if layout == "twin":
            # Layout con doppio asse Y (WP a sinistra, PSNR a destra)
            fig, ax1 = plt.subplots(figsize=(8, 5))

            ax1.plot(mu_vals, W, label="RWP", color="tab:blue")
            ax1.plot(mu_best_wp, W[min_idx], 'go')
            ax1.yaxis.set_major_formatter(formatter)
            ax1.set_xlabel("$\mu$")
            ax1.set_ylabel("$W(\mu)$", color="tab:blue")
            ax1.tick_params(axis='y', labelcolor="tab:blue")
            ax1.axvline(mu_best_wp.item(), color="green", linestyle="--", alpha=0.7)

            ax2 = ax1.twinx()
            ax2.plot(mu_vals, psnr, label="PSNR", color="tab:orange")
            ax2.plot(mu_vals[max_idx_psnr], psnr[max_idx_psnr], 'ro')
            ax2.set_ylabel("PSNR (dB)", color="tab:orange")
            ax2.tick_params(axis='y', labelcolor="tab:orange")
            ax2.axvline(mu_vals[max_idx_psnr].item(), color="red", linestyle="--", alpha=0.7)

            # Combina le legende
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

            plt.title(title)
            plt.tight_layout()
            plt.grid(True)
            plt.show()

        elif layout == "stacked":
            # Layout con 3 subplot separati verticalmente
            fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

            # RWP
            axs[0].plot(mu_vals, W, color="tab:blue")
            axs[0].plot(mu_best_wp, W[min_idx], 'go')
            axs[0].axvline(mu_best_wp.item(), color="green", linestyle="--")
            axs[0].yaxis.set_major_formatter(formatter)
            axs[0].set_ylabel("WP")
            axs[0].grid(True)

            # PSNR
            axs[1].plot(mu_vals, psnr, color="tab:orange")
            axs[1].plot(mu_vals[max_idx_psnr], psnr[max_idx_psnr], 'ro')
            axs[1].axvline(mu_vals[max_idx_psnr].item(), color="red", linestyle="--")
            axs[1].set_ylabel("PSNR (dB)")
            axs[1].grid(True)

            # SSIM
            axs[2].plot(mu_vals, ssim, color="tab:green")
            axs[2].plot(mu_vals[max_idx_ssim], ssim[max_idx_ssim], 'go')
            axs[2].axvline(mu_vals[max_idx_ssim].item(), color="green", linestyle="--")
            axs[2].set_ylabel("SSIM")
            axs[2].set_xlabel("$\mu$")
            axs[2].grid(True)

            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
                       
    plot_results(results_best, dataset, is_real, is_3d, pxsize, x0_sec = 100, y0_sec = 100)