import deepinv as dinv
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS

psnr = PSNR()
mse = MSE()
# ();lpips = LPIPS();()
ssim = SSIM()

import torch
import torch.nn.functional as F


def fuzzy_jaccard_map(pred, gt):
    """
    Calcola il Fuzzy Jaccard pixel per pixel tra due immagini continue.
    
    Args:
        pred (torch.Tensor): Immagine predetta con valori in [0, 1].
        gt (torch.Tensor): Ground Truth con valori in [0, 1].
        
    Returns:
        torch.Tensor: Mappa dello stesso formato, con valori da 0.0 a 1.0
                      che rappresentano la percentuale di match per ogni pixel.
    """
    eps = 1e-5
    # 1. Calcoliamo minimo e massimo pixel per pixel
    min_vals = torch.minimum(pred, gt) + eps
    max_vals = torch.maximum(pred, gt) + eps
    
    # 2. Creiamo una maschera per trovare i pixel dove ENTRAMBI valgono 0
    # (cioè dove il valore massimo è 0)
    both_zero_mask = (max_vals == 0.0)
    
    # 3. Evitiamo la divisione per zero sostituendo temporaneamente gli zeri 
    # nel denominatore con 1.0
    safe_max = max_vals.masked_fill(both_zero_mask, 1.0)
    
    # 4. Calcoliamo il rapporto
    jaccard_map = min_vals / safe_max
    
    # 5. Imponiamo manualmente il punteggio perfetto (1.0) dove entrambi erano zero
    jaccard_map = jaccard_map.masked_fill(both_zero_mask, 1.0)
    
    return jaccard_map


def exponential_similarity_map(pred, gt, alpha=5.0):
    """
    Trasforma l'errore in una percentuale di similarità.
    alpha = 5.0 è un buon valore: un errore di 0.2 dimezza il punteggio (0.36)
    """
    return torch.exp(-alpha * torch.abs(pred - gt))


def binary_error_map(pred, gt, threshold=0.1):
    """
    Crea una mappa RGB per visualizzare gli errori della binarizzazione.
    Restituisce un tensore di forma (3, H, W).
    """
    pred_mask = (pred > threshold).float().squeeze()
    gt_mask = (gt > threshold).float().squeeze()
    
    # Inizializziamo i tre canali RGB a zero (tutto nero)
    H, W = pred_mask.shape
    rgb_map = torch.zeros((3, H, W), device=pred.device)
    
    # Canale 0 (Rosso), Canale 1 (Verde), Canale 2 (Blu)
    
    # TRUE POSITIVES (Bianco: R=1, G=1, B=1)
    true_pos = (pred_mask * gt_mask).bool()
    rgb_map[0, true_pos] = 1.0
    rgb_map[1, true_pos] = 1.0
    rgb_map[2, true_pos] = 1.0
    
    # FALSE POSITIVES - Allarmi falsi (Rosso: R=1, G=0, B=0)
    false_pos = (pred_mask > gt_mask).bool()
    rgb_map[0, false_pos] = 1.0
    
    # FALSE NEGATIVES - Filamenti mancati (Blu: R=0, G=0, B=1)
    false_neg = (gt_mask > pred_mask).bool()
    rgb_map[2, false_neg] = 1.0
    
    return rgb_map


def classic_jaccard_score(pred, gt, threshold=0.1):
    """
    Calcola il Jaccard Index (IoU) classico binarizzando le immagini.
    
    Args:
        pred: Tensore della predizione (media a posteriori)
        gt: Tensore della Ground Truth
        threshold: Valore sopra il quale il pixel diventa 1 (filamento)
    """
    # 1. Binarizzazione: Trasformiamo in 0.0 o 1.0
    pred_mask = (pred > threshold).float()
    gt_mask = (gt > threshold).float()
    
    # 2. Intersezione (AND logico: pixel accesi in entrambe le maschere)
    intersection = (pred_mask * gt_mask).sum()
    
    # 3. Unione (OR logico: pixel accesi in almeno una delle due maschere)
    union = pred_mask.sum() + gt_mask.sum() - intersection
    
    # 4. Gestione sicura del background vuoto
    if union == 0:
        return 1.0  # Se non ci sono filamenti né in GT né in Pred, il match è perfetto
        
    iou = intersection / union
    return iou.item()



def local_binary_jaccard_map(pred, gt, threshold=0.1, window_size=5):
    """
    Calcola il Jaccard binario su una finestra (patch) attorno a ogni pixel.
    window_size: grandezza del quadratino (es. 5x5). Più è grande, più la mappa è sfumata.
    """
    pred_mask = (pred > threshold).float()
    gt_mask = (gt > threshold).float()
    
    # Se i tensori non hanno i canali, li aggiungiamo per poter usare l'AvgPool2d (B, C, H, W)
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
        
    # Calcoliamo Intersezione e Unione globali
    intersection = pred_mask * gt_mask
    union = torch.max(pred_mask, gt_mask) # Equivale a pred_mask + gt_mask - intersection
    
    # Sommiamo i pixel all'interno della finestra usando l'Average Pooling
    # padding=window_size//2 serve a mantenere l'immagine della stessa grandezza originale
    local_intersection = F.avg_pool2d(intersection, kernel_size=window_size, stride=1, padding=window_size//2)
    local_union = F.avg_pool2d(union, kernel_size=window_size, stride=1, padding=window_size//2)
    
    # Calcoliamo il Jaccard locale, evitando divisioni per zero
    local_jaccard = local_intersection / (local_union + 1e-8)
    
    # Forziamo a 1.0 le zone dove l'unione locale è zero (tutto sfondo corretto)
    local_jaccard = torch.where(local_union == 0, torch.tensor(1.0, device=pred.device), local_jaccard)
    
    return local_jaccard.squeeze() # Togliamo le dimensioni extra prima di restituire



def colorbar_for_image(tensor, cmap, caption=""):
# Converte il tensore in numpy
    data = tensor.cpu().squeeze().numpy()
    
    # Usa un context manager per disabilitare LaTeX SOLO per questo grafico
    with plt.rc_context({'text.usetex': False}):
        # Crea la figura e l'asse
        fig, ax = plt.subplots()
        
        # Disegna l'immagine con il colormap 'hot'
        im = ax.imshow(data, cmap=cmap)
        
        # Aggiungi la colorbar affiancata
        fig.colorbar(im, ax=ax)
        
        # Nascondi gli assi principali
        ax.axis('off')
        
        # Crea l'oggetto wandb.Image passando la figura
        wandb_img = wandb.Image(fig, caption=caption)
        
        # Chiudi la figura per evitare memory leak
        plt.close(fig)
        
    return wandb_img