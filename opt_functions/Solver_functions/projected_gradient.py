import torch
import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS
from .metrics import *
from tqdm import tqdm
# from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity
import torch
import math
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssim = SSIM()

def tresholding (x_input, lam, tau):
    T = torch.sign(x_input) * torch.max(torch.abs(x_input)-lam*tau,torch.tensor(0))
    return T

def Bregman_h(x, u, eps = 1e-8):
    log_term = -torch.log(x + eps) + torch.log(u + eps)
    term1 = torch.sum(log_term, dim=(1, 2, 3))
    dot_term = (-1 / (u + eps)) * (x - u)
    term2 = torch.sum(dot_term, dim=(1, 2, 3))
    return term1 - term2

def crop_center(img, cropx, cropy):
    y, x = img.shape[2], img.shape[3]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:,:,starty : starty + cropy, startx : startx + cropx]


def identity(x, sigma):
    return x


class BaseISMSolver:
    """
    Gestisce il ciclo, le metriche e unifica la matematica PGD/PROX.
    Le sottoclassi implementeranno solo il _step() specifico.
    """
    def __init__(self, parameters, algorithm="pgd", is_3d=False, is_realdata=False):
        self.params = parameters
        self.algorithm = algorithm.lower()
        self.is_3d = is_3d
        self.is_realdata = is_realdata
        self.device = parameters["x_init"].device
        self.back = parameters["back"].device
        
        self.max_iter = parameters["max_iter"]
        self.lam = parameters["lam"]
        self.L_max = parameters["Lip_reg"]
        self.tollerance = parameters["tollerance"]
        self.physics = parameters["physics"]
        self.data_fid = parameters["data_fid"]
        self.grad_data_fid = parameters["grad_data_fid"]
        self.prior = parameters["prior"]

    def _get_candidate_and_metrics(self, y_eval, tau, y, lam, precomputed_g_data=None, precomputed_f_y_eval=None):
            # Se non li passiamo, li calcoliamo (come prima)
            g_data = self.grad_data_fid(y, y_eval, self.physics) if precomputed_g_data is None else precomputed_g_data
            
            if self.algorithm == "pgd":
                g_prior = lam * self.params["prior_grad"](y_eval)
                g_tot = g_data + g_prior
                x_next = torch.clamp(y_eval - tau * g_tot, min=0.0)
                
                f_y_eval = (self.data_fid(y, y_eval, self.physics) + lam * self.prior(y_eval)) if precomputed_f_y_eval is None else precomputed_f_y_eval
                f_x_next = self.data_fid(y, x_next, self.physics) + lam * self.prior(x_next)

                g_for_dot = g_tot
                
            elif self.algorithm == "prox":
                x_next = torch.clamp(self.params["prox"](y_eval - tau * g_data, lam, tau), min=0.0)
                
                f_y_eval = self.data_fid(y, y_eval, self.physics) if precomputed_f_y_eval is None else precomputed_f_y_eval
                f_x_next = self.data_fid(y, x_next, self.physics)
                g_for_dot = g_data
                
                
            elif self.algorithm == "md":
                
                g_prior = lam * self.params["prior_grad"](y_eval)
                g_tot = g_data + g_prior
                x_next = y_eval / (1 + tau * y_eval * g_tot)
                
                f_y_eval = (self.data_fid(y, y_eval, self.physics) + lam * self.prior(y_eval)) if precomputed_f_y_eval is None else precomputed_f_y_eval
                f_x_next = self.data_fid(y, x_next, self.physics) + lam * self.prior(x_next)

                g_for_dot = g_tot
                
            
            else:
                raise ValueError(f"Algoritmo {self.algorithm} non supportato.")
            
            return x_next, g_for_dot, f_y_eval, f_x_next


    def _step(self, state, y, lam, back):
        raise NotImplementedError("Scegli un solver specifico (es. Fast_Solver_Backtracking)")

    def solve(self, y):
        state = {
            'x_curr': self.params["x_init"].clone(),
            'x_prev': self.params["x_init"].clone(),
            'tau_k': 1.0 / self.L_max,
            't_k': 1.0
        }
        
        funct = torch.zeros(self.max_iter, device=self.device)
        iter_err = torch.zeros(self.max_iter, device=self.device)
        
        if not self.is_realdata:
            diff_fid = torch.zeros(self.max_iter, device=self.device)
            psnr_vec = torch.zeros(self.max_iter, device=self.device)
            ssim_vec = torch.zeros(self.max_iter, device=self.device)

        for k in tqdm(range(self.max_iter), desc=f"iter_{self.__class__.__name__}_{self.algorithm.upper()}"):
            
            # --- CHIAMATA AL METODO SPECIFICO DEL SOLVER ---
            state = self._step(state, y)
            x_next = state['x_curr']
            x_prev = state['x_prev'] 

            # --- METRICHE ---
            if 'f_x_next' in state:
                funct[k] = state['f_x_next']
            else:
                funct[k] = self.data_fid(y, x_next, self.physics) + self.lam * self.prior(x_next)
            
            x_prev_err = x_prev[:, 1:2] if self.is_3d else x_prev
            x_next_err = x_next[:, 1:2] if self.is_3d else x_next
            iter_err[k] = torch.norm(x_prev_err - x_next_err, 'fro') / (torch.norm(x_prev_err, 'fro') + 1e-10)

            if not self.is_realdata:
                x_gt = self.params["ground_truth"]
                x_gt_norm = (x_gt[:, 1:2] if self.is_3d else x_gt) / (x_gt[:, 1:2] if self.is_3d else x_gt).max()
                x_next_norm = (x_next_err if self.is_3d else x_next) / (x_next_err if self.is_3d else x_next).max()

                diff_fid[k] = self.params["single_data_fid"](x_gt_norm, x_next_norm)
                psnr_vec[k] = psnr(x_gt_norm, x_next_norm)
                ssim_vec[k] = ssim(x_gt_norm, x_next_norm)

            # --- CONVERGENZA ---
            if iter_err[k] < self.tollerance:
                print(f"Convergence reached at iter = {k}, lambda = {self.lam}")
                funct, iter_err = funct[:k], iter_err[:k]
                if not self.is_realdata:
                    diff_fid, psnr_vec, ssim_vec = diff_fid[:k], psnr_vec[:k], ssim_vec[:k]
                break

        return {'x_result': state['x_curr'], 'funct': funct, 'iter_err': iter_err,
                'diff_fid': None if self.is_realdata else diff_fid,
                'psnr': None if self.is_realdata else psnr_vec,
                'ssim': None if self.is_realdata else ssim_vec}


# ==========================================
# 2. I 4 SOLVER ESPLICITI (La Dinamica)
# ==========================================

class Pgd(BaseISMSolver):
    def _step(self, state, y):
        x_curr = state['x_curr']
        tau = 1.0 / self.L_max
        x_next, _, _, f_x_next = self._get_candidate_and_metrics(x_curr, tau, y, self.lam)
        
        state['x_prev'] = x_curr
        state['x_curr'] = x_next
        state['f_x_next'] = f_x_next # Salviamo la loss!
        return state

class Pgd_Backtracking(BaseISMSolver):
    # Aggiungiamo l'init per avere delta, s ed eta come attributi di classe
    def __init__(self, parameters, algorithm="pgd", is_3d=False, is_realdata=False, s=1.0, eta=2, delta=0.9):
        super().__init__(parameters, algorithm, is_3d, is_realdata)
        self.s = s
        self.eta = eta
        self.delta = delta

    def _step(self, state, y):
        x_curr = state['x_curr']
        tau_k = state['tau_k'] # Recuperiamo il passo precedente!
        
        # Rendiamo L_candidate adattivo (esattamente come nel Fast)
        tau_candidate = min(tau_k / self.delta, 1.0 / self.s)
        L_candidate = 1.0 / tau_candidate
        
        # PRE-CALCOLIAMO GRADIENTE E LOSS DI x_curr FUORI DAL CICLO!
        pre_g_data = self.grad_data_fid(y, x_curr, self.physics)
        if self.algorithm == "pgd":
            pre_f_y_eval = self.data_fid(y, x_curr, self.physics) + self.lam * self.prior(x_curr)
        else:
            pre_f_y_eval = self.data_fid(y, x_curr, self.physics)

        loop_count = 0
        while True:
            tau = 1.0 / L_candidate
            
            x_next, g_for_dot, f_y_eval, f_x_next = self._get_candidate_and_metrics(
                x_curr, tau, y, self.lam, precomputed_g_data=pre_g_data, precomputed_f_y_eval=pre_f_y_eval
            )
            
            diff_x = x_next - x_curr
            grad_dot = torch.sum(g_for_dot * diff_x)
            
            sqnorm = torch.sum(diff_x ** 2)
            
            if self.algorithm == "pgd":
                
                dist  = grad_dot + (L_candidate / 2.0) * sqnorm
                
            elif self.algorithm == "md":
            
                dist = - (0.8/tau)*Bregman_h(x_next, x_curr)
        
            
            if f_x_next <= f_y_eval + grad_dot + dist:
                break
                
            L_candidate *= self.eta
            loop_count += 1
            if L_candidate >= self.L_max or loop_count > 50:
                L_candidate = self.L_max
                tau = 1.0 / L_candidate
                x_next, _, _, f_x_next = self._get_candidate_and_metrics(
                    x_curr, tau, y, self.lam, precomputed_g_data=pre_g_data, precomputed_f_y_eval=pre_f_y_eval
                )
                break
                
        state['x_prev'] = x_curr
        state['x_curr'] = x_next
        state['tau_k'] = tau          # SALVIAMO IL NUOVO PASSO per la prossima iterazione!
        state['f_x_next'] = f_x_next  # Salviamo la loss!
        return state


class Pgd_Fast(BaseISMSolver):
    """Accelerazione di Nesterov (FISTA), nessun backtracking."""
    
    def _step(self, state, y):
        x_curr = state['x_curr']
        x_prev = state['x_prev']
        t_k = state['t_k']
        tau = 1.0 / self.L_max
        
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * (t_k ** 2))) / 2.0
        beta = (t_k - 1.0) / t_next
        y_eval = x_curr + beta * (x_curr - x_prev)
        
        x_next, _, _, f_x_next = self._get_candidate_and_metrics(y_eval, tau, y, self.lam)
        
        state['x_prev'] = x_curr
        state['x_curr'] = x_next
        state['t_k'] = t_next
        state['f_x_next'] = f_x_next # Salviamo la loss!
        return state
    
    
class Pgd_Fast_Backtracking(BaseISMSolver):
    def __init__(self, parameters, algorithm="pgd", is_3d=False, is_realdata=False, s=1.0, eta=2, delta=0.9):
        super().__init__(parameters, algorithm, is_3d, is_realdata)
        self.s = s
        self.eta = eta
        self.delta = delta

    def _step(self, state, y):
        x_curr = state['x_curr']
        x_prev = state['x_prev']
        tau_k = state['tau_k']
        t_k = state['t_k']
        
        tau_candidate = min(tau_k / self.delta, 1.0 / self.s)
        L_candidate = 1.0 / tau_candidate
        
        loop_count = 0
        while True:
            tau = 1.0 / L_candidate
            
            # Calcolo inerzia
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * (tau_k / tau) * (t_k ** 2))) / 2.0
            beta = (t_k - 1.0) / t_next
            
            # 1. FIX FISICO: Proiettiamo l'estrapolazione sul dominio positivo
            # Evita che y_eval negativo faccia esplodere i logaritmi della KL
            y_eval = torch.clamp(x_curr + beta * (x_curr - x_prev), min=0.0)
            
            # Richiesta passo e metriche (PGD o PROX)
            x_next, g_for_dot, f_y_eval, f_x_next = self._get_candidate_and_metrics(y_eval, tau, y, self.lam)
            
            # Controllo discesa
            diff_x = x_next - y_eval
            grad_dot = torch.sum(g_for_dot * diff_x)
            sqnorm = torch.sum(diff_x ** 2)
            
            if f_x_next <= f_y_eval + grad_dot + (L_candidate / 2.0) * sqnorm:
                break
                
            # Restringi passo
            L_candidate *= self.eta
            loop_count += 1
            
            if L_candidate >= self.L_max or loop_count > 50:
                L_candidate = self.L_max
                tau = 1.0 / L_candidate
                
                t_next = (1.0 + math.sqrt(1.0 + 4.0 * (tau_k / tau) * (t_k ** 2))) / 2.0
                beta = (t_k - 1.0) / t_next
                y_eval = torch.clamp(x_curr + beta * (x_curr - x_prev), min=0.0) # Clamp anche qui
                
                x_next, _, _, f_x_next = self._get_candidate_and_metrics(y_eval, tau, y, self.lam)
                break
                
        # 2. FIX MATEMATICO: RESTART ADATTIVO
        # Controlliamo se stiamo oscillando: se la direzione dell'aggiornamento
        # è opposta a quella dell'inerzia, abbiamo superato la buca!
        if torch.sum((x_curr - x_prev) * (x_next - x_curr)) < 0:
            t_next = 1.0  # Azzeriamo l'inerzia per la prossima iterazione
            state['x_prev'] = x_next # Dimentichiamo il passato
        else:
            state['x_prev'] = x_curr

        state['x_curr'] = x_next
        state['tau_k'] = tau
        state['t_k'] = t_next
        state['f_x_next'] = f_x_next
        
        return state