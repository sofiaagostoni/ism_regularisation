import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import scipy.optimize as opt
import torch
from sklearn.metrics import r2_score

from tqdm import trange
import multiprocessing
from joblib import Parallel, delayed

from . import APR_lib as APR

#%%

class Selector:
    
    def __init__(self, img):

        self.fig, self.ax = plt.subplots()
        m = self.ax.imshow(img)
        plt.colorbar( m, ax = self.ax)
        plt.axis('off')
        self.ax.set_title('Drag a region with mouse')
        
        self.coord = None
        self.rs = None
        
        self.select()
        
        while self.coord is None:
            plt.pause(1)

    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        self.coord = torch.asarray([ [x1, y1], [x2, y2] ]).astype(int)
    
        rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), torch.abs(x1-x2), torch.abs(y1-y2), fill = False )
        self.ax.add_patch(rect)
    
        self.rs.set_visible(False)
        self.rs.set_active(False)

    def select(self):
        self.rs = RectangleSelector(self.ax, self.line_select_callback,
                       useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=False)

#%%

def fingerprint_model(xdata, A, sigma, BKG = 0):
    x = xdata[0,:]
    y = xdata[1,:]
    return A * torch.exp( -( (x)**2 + (y)**2 ) / (2*sigma**2) ) + BKG

def gaussian(x, y, x0, y0, A, sigma):
    Gauss = torch.exp( -( (x-x0)**2 + (y-y0)**2 ) / (2*sigma**2) )
    return A * Gauss  / torch.sum(Gauss)

def gaussian_model(xdata, sigma_A, B, sigma_B):
    x = xdata[0,:]
    y = xdata[1,:]
    # sigma_A = 1.3*( 0.83 / torch.sqrt(2*torch.log(2)) )
    A = 1 - B
    return gaussian(x, y, 0, 0, A, sigma_A) + gaussian(x, y, 0, 0, B, sigma_B)

#%%

def FitFingerprint( sub_img, p0 = [1,1], Ndet = 5):
    
    # Calculate fingerprint
    
    fingerprint = torch.sum( sub_img, axis = (0,1) ).reshape(Ndet, Ndet)
    fingerprint = fingerprint / torch.max(fingerprint)
    
    # The two-dimensional domain of the fit.
    
    x = torch.linspace( -(Ndet//2), Ndet//2, Ndet)
    X, Y = torch.meshgrid(x, x)

    xdata = torch.vstack( ( X.ravel(), Y.ravel() ) ) 

    # Fit fingerprint with a single gaussian function

    popt, pcov = opt.curve_fit( fingerprint_model, xdata, fingerprint.ravel(), p0 )

    A = popt[0]
    sigma_fing = popt[1]
    BKG = 0

    fit_fingerprint = fingerprint_model(xdata, A, sigma_fing, BKG).reshape(Ndet, Ndet)
    
    return fingerprint, fit_fingerprint, popt

#%%

def pixel_fit_2(F, sigma_A, sigma_B_bound = None, threshold = 0):
    '''
    It fits the input micro-image to the sum of two Gaussian functions.
    The in-focus curve has fixed mean (the center of the micro-image)
    and standard deviation (sigma_A). The out-of-focus curve has fixed mean
    (the center of the micro-image) and free standard deviation (sigma_B), with
    a lower bound sigma_B_bound.

    Parameters
    ----------
    F : torch.tensor
        Micro-image array.
    sigma_A : float
        Standard deviation of the in-focus Gaussian function (units of pixels).
    sigma_B_bound : float, optional
        Lower limit of the background std, in units of in-focus std.
        The default is None.
    threshold : int, optional
        Minimum number of photons per pixel required to start
        the analysis. Pixels below the threshold are assigned
        to the background.. The default is 0.

    Returns
    -------
    bkg : TYPE
        Background micro-image.
    sig : TYPE
        In-focus micro-image.
    sigma_B : TYPE
        Fitted sigma_B value. If the fit was unsuccesful or the treshold
        criterium is not satisfied, a 0 is returned.
    R2 : TYPE
        Goodness of fit value (R-squared).
    '''
    
    if sigma_B_bound is None:
        sigma_B_bound = 3*sigma_A

    N = torch.sum(F) # Normalization factor - sum of all pixels
    
    if N > threshold: # Thresholding
        
        F = F / N
        
        # The two-dimensional domain of the fit.
        
        xmin, xmax, nx = -2, 2, 5
        ymin, ymax, ny = -2, 2, 5
        x, y = torch.linspace(xmin, xmax, nx), torch.linspace(ymin, ymax, ny)
        X, Y = torch.meshgrid(x, y)
        
        # Ravel the meshgrids of X, Y points to a pair of 1-D arrays.
        
        xdata = torch.vstack((X.ravel(), Y.ravel())) 

        # Perform fitting
        
        p0 = [0.1, 4*sigma_A]
        fit_model = lambda xdata, B, sigma_B: gaussian_model(xdata, sigma_A, B, sigma_B)
        
        try:
            popt, pcov = opt.curve_fit(fit_model, xdata, F, p0, bounds = ([0, sigma_B_bound], [1, torch.inf]))
            A = 1 - popt[0]
            B = popt[0]
            sigma_B = popt[1]
        except:
            bkg = F
            sig = torch.zeros(F.shape)
            sigma_B = 0
            R2 = 0
            return bkg, sig, sigma_B, R2
        
        sig = N * gaussian(X,Y, 0, 0, A, sigma_A).ravel()
        bkg = N * gaussian(X,Y, 0, 0, B, sigma_B).ravel()
            
        R2 = r2_score(F*N, sig+bkg)
    else:
        
        bkg = F
        sig = torch.zeros(F.shape)
        sigma_B = 0
        R2 = 0
        
    return bkg, sig, sigma_B, R2


def pixel_fit_1(F, sigma_A, sigma_B, threshold = 0):
    '''
    It fits the input micro-image to the sum of two Gaussian functions.
    The in-focus curve has fixed mean (the center of the micro-image)
    and standard deviation (sigma_A). The out-of-focus curve has fixed mean
    (the center of the micro-image) and fixed standard deviation (sigma_B).

    Parameters
    ----------
    F : torch.tensor
        Micro-image array.
    sigma_A : float
        Standard deviation of the in-focus Gaussian function (units of pixels).
    sigma_B : float
        Standard deviation of the ou-of-focus Gaussian function (units of pixels).
    threshold : int, optional
        Minimum number of photons per pixel required to start
        the analysis. Pixels below the threshold are assigned
        to the background.. The default is 0.

    Returns
    -------
    bkg : TYPE
        Background micro-image.
    sig : TYPE
        In-focus micro-image.
    sigma_B : TYPE
        Fitted sigma_B value. If the fit was unsuccesful or the treshold
        criterium is not satisfied, a 0 is returned.
    R2 : TYPE
        Goodness of fit value (R-squared).
    '''
    
    N = torch.sum(F) # Normalization factor - sum of all pixels

    if N > threshold: # Thresholding
        
        F = F / N
        
        # The two-dimensional domain of the fit.
        
        xmin, xmax, nx = -2, 2, 5
        ymin, ymax, ny = -2, 2, 5
        x, y = torch.linspace(xmin, xmax, nx), torch.linspace(ymin, ymax, ny)
        X, Y = torch.meshgrid(x, y)
        
        # Ravel the meshgrids of X, Y points to a pair of 1-D arrays.
        
        xdata = torch.vstack((X.ravel(), Y.ravel())) 
    
        # Perform fitting
        
        p0 = [0.5]
        fit_model = lambda xdata, B: gaussian_model(xdata, sigma_A, B, sigma_B)
        popt, pcov = opt.curve_fit(fit_model, xdata, F, p0, bounds = (0, [1]))
        
        # Classify signal and background
        
        A = 1 - popt[0]
        B = popt[0]
    
        sig = N * gaussian(X,Y, 0, 0, A, sigma_A).ravel()
        bkg = N * gaussian(X,Y, 0, 0, B, sigma_B).ravel()
            
        R2 = r2_score(F*N, sig+bkg)
    else:
        
        bkg = F
        sig = torch.zeros(F.shape)
        R2 = 0
        
    return bkg, sig, R2

#%%

def focusISM(img, sigma_B_bound = None, threshold = 0, apr = True, calibration = 'manual', sum_results = True, parallelize = True):
    """
    Focus-ISM algorithm to remove out-of-focus background

    Parameters
    ----------
    img : torch.array (Nx x Ny x Nch)
        ISM dataset
    sigma_B_bound : float
        lower limit of the background std, in units of in-focus std
    threshold : int
        Minimum number of photons per pixel required to start
        the analysis. Pixels below the threshold are assigned
        to the background.
    apr : bool
        If True, the ISM dataset is reassigned with APR before
        applying focus-ISM. This step is facultative only for
        high-power STED data.
    calibration : str or torch.array(Nx x Ny x Nch)
        if 'manual' the user is requested to select a region
        of the input dataset. If torch.array(Nx x Ny x Nch), the
        calibration dataset is used to calculate the in-focus fingerprint
    sum_results : bool
        If true, the results are summed along the Nch dimension
    parallelize : bool
        If True, the algorithm is CPU-parallelized using the 'threading' backend.
        If False, a progress bar is displayed.
        Default is True.

    Returns
    -------
    signal : torch.array (Nx x Ny) or torch.array (Nx x Ny x Nch)
        Focus-ISM reconstruction of the in-focus signal
    background : torch.array (Nx x Ny) or torch.array (Nx x Ny x Nch)
        Focus-ISM reconstruction of the out-of-focus signal,
    ism : torch.array (Nx x Ny) or torch.array (Nx x Ny x Nch)
        APR reconstruction
    """
        
    sz = img.shape
    Ndet = torch.sqrt(sz[-1]).astype(int)

    usf = 10
    ref = Ndet**2//2
    
    # Adaptive pixel reassignment
    
    if apr == True:
        shift, img_ism = APR.APR(img, usf, ref, filter_sigma=1)
        img_ism[img_ism<0] = 0
    else:
        img_ism = img
    
    # Calibrate algorithm with fingerprint
    
    ism_sum = torch.sum(img_ism, axis = 2)

    if isinstance(calibration, str) and calibration == 'manual':
        xy = Selector( ism_sum ).coord
        subimg = img[ xy[0,1]:xy[1,1], xy[0,0]:xy[1,0], :]
    else:
        subimg = calibration

    fingerprint, fit, popt = FitFingerprint( subimg )
    sigma_fing = popt[1]
    
    # Fit each micro-image with two Gaussians

    sz = img.shape
    img_reshaped = img_ism.reshape(sz[0]*sz[1], sz[2])

    N = multiprocessing.cpu_count() - 1

    sigma_A = sigma_fing
    
    if sigma_B_bound is None:
        sigma_B_bound = 2*sigma_A
    else:
        sigma_B_bound *= sigma_A
        
    threshold = threshold

    if parallelize == True:
        print('Focus-ISM - parallel:')
        Result = Parallel(n_jobs = N, backend = 'threading')( delayed(pixel_fit_2)( img_reshaped[i,:], sigma_A, sigma_B_bound = sigma_B_bound, threshold = threshold) for i in trange(sz[0]*sz[1]) )
    else:
        Result =[[]] * (sz[0] * sz[1])
        print('Focus-ISM:')
        for i in trange(sz[0] * sz[1]):
            Result[i] = pixel_fit_2(img_reshaped[i, :], sigma_A, sigma_B_bound = sigma_B_bound, threshold = threshold)

    # Reshape
    
    bkg = torch.asarray([Result[j][0] for j in range(len(Result))]).reshape(sz[0], sz[1], sz[2])
    
    sig = torch.asarray([Result[j][1] for j in range(len(Result))]).reshape(sz[0], sz[1], sz[2])
    
    if sum_results == True:
    
        Bkg_sum = torch.sum(bkg, axis = 2)
        
        Focus_sum = torch.sum(sig, axis = 2)
        
        return Focus_sum, Bkg_sum, ism_sum
    
    else:
        
        return sig, bkg, img_ism