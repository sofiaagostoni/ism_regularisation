import numpy as np
import torch

from .FRC_lib import radial_profile

def sigmoid(R: float, T: float, S: float):
    '''
    It generates a circularly-symmetric sigmoid function.

    Parameters
    ----------
    R : float
        Radial axis.
    T : float
        Cut-off frequency.
    S : float
        Sigmoid slope.

    Returns
    -------
    torch.tensor
        Sigmoid array. It has the same dimensions of R.

    '''

    return 1 / (1 + torch.exp((R - T) / S))


def low_pass(img: torch.tensor, T: float, S: float, data: str = 'real'):
    '''
    It applies a low-pass sigmoidal filter to a 2D image.

    Parameters
    ----------
    img : torch.tensor
        2D image.
    T : float
        Cut-off frequency.
    S : float
        Sigmoid slope.
    data : str, optional
        Domain of the image: It can be 'real' or 'fourier'.
        The default is 'real'.

    Returns
    -------
    img_filt : torch.tensor
        Filtered 2D image, in the domain specified by 'data'.

    '''

    if data == 'real':
        img_fft = torch.fft.fftn(img, axes=(0, 1))
        img_fft = torch.fft.fftshift(img_fft, axes=(0, 1))
    elif data == 'fourier':
        img_fft = img
    else:
        raise ValueError('data has to be \'real\' or \'fourier\'')

    Nx = torch.shape(img_fft)[0]
    Ny = torch.shape(img_fft)[1]
    cx = int((Nx + torch.mod(Nx, 2)) / 2)
    cy = int((Ny + torch.mod(Ny, 2)) / 2)

    x = (torch.arange(Nx) - cx) / Nx
    y = (torch.arange(Ny) - cy) / Ny

    X, Y = torch.meshgrid(x, y)
    R = torch.sqrt(X ** 2 + Y ** 2)

    sig = sigmoid(R, T, S)

    img_filt = torch.einsum('ij..., ij -> ij...', img_fft, sig)

    if data == 'real':
        img_filt = torch.fft.ifftshift(img_filt, axes=(0, 1))
        img_filt = torch.fft.ifftn(img_filt, axes=(0, 1))
        img_filt = torch.abs(img_filt)

    return img_filt


# %%

import torch

def Reorder(dset, inOrder: str, outOrder: str = 'rzxytc'):
    '''
    Reorders a dataset to match the desired order of dimensions.
    If some dimensions are missing, it adds new dimensions.
    '''
    
    data = dset.clone()
    Nout = len(outOrder)
    Ndim = len(inOrder)

    if Ndim < Nout:
        # --- CASO 1: Espansione (Aggiungere dimensioni mancanti) ---
        
        # Lista temporanea per tracciare l'ordine attuale delle dimensioni dei dati
        current_order = list(inOrder)
        
        # 1. Aggiungiamo le dimensioni mancanti tutte in fondo (unsqueeze(-1))
        #    È il modo più sicuro per non alterare gli indici esistenti.
        for char in outOrder:
            if char not in current_order:
                data = data.unsqueeze(-1)
                current_order.append(char)
        
        # 2. Ora 'data' ha tutte le dimensioni necessarie, ma sono in ordine sparso 
        #    (quelle originali prima, quelle nuove in fondo).
        #    Calcoliamo la permutazione necessaria per ottenere 'outOrder'.
        permute_indices = [current_order.index(char) for char in outOrder]
        
        data = data.permute(permute_indices)

    else:
        # --- CASO 2: Riduzione (Rimuovere dimensioni in eccesso) ---
        
        # 1. Creiamo le slice per rimuovere le dimensioni non volute.
        #    Se una dimensione di inOrder non è in outOrder, prendiamo l'indice 0.
        slices = []
        remaining_dims_chars = [] # Tracciamo quali lettere rimangono
        
        for char in inOrder:
            if char in outOrder:
                slices.append(slice(None)) # Equivale a ':'
                remaining_dims_chars.append(char)
            else:
                slices.append(0) # Equivale a prendere l'indice 0 ed eliminare la dim
        
        # Applichiamo il taglio
        data = data[tuple(slices)]
        
        # 2. Ora riordiniamo le dimensioni rimaste per matchare outOrder
        permute_indices = [remaining_dims_chars.index(char) for char in outOrder]
        
        data = data.permute(permute_indices)

    return data

# def Reorder(dset, inOrder: str, outOrder: str = 'rzxytc'):
#     '''
#     It reorders a dataset to match the desired order of dimensions.
#     If some dimensions are missing, it adds new dimensions.

#     Parameters
#     ----------
#     dset : tensor
#         ISM dataset.
#     inOrder : str
#         Order of the dimension of the data.
#         It can contain any letter of the outOrder string.
#     outOrder : str, optional
#         Order of the output. The default is 'rzxytc'.

#     Returns
#     -------
#     data : tensor
#         ISM dataset reordered.

#     '''

#     data = dset.clone()

#     Nout = len(outOrder)
#     Ndim = len(inOrder)

#     if (Ndim < Nout):
#         # check where the current dimensions are located
#         idx = torch.empty( Nout )
#         for n, c in enumerate(outOrder):
#             idx[n] = outOrder.find(c)
#         idx = idx.astype('int')

#         # add missing dimensions
#         slices = []
#         for i in idx:
#             if i == -1:
#                 slices.append(torch.newaxis)
#             else:
#                 slices.append(torch.s_[:])

#         slices = tuple(slices)
#         data = data[slices]

#         # reorder final dimensions
#         idx2 = torch.empty( Ndim )
#         for n, c in enumerate(inOrder):
#             idx2[n] = torch.char.find(outOrder, c)
#         idx2 = idx2.astype('int')

#         order = idx.copy()
#         order[torch.where(idx != -1)] = idx2
#         order[torch.where(idx == -1)] = torch.argwhere(idx == -1).flatten()

#         data = torch.moveaxis(data, torch.arange(Nout), order)

#     else:
#         # check where the dimensions are located
#         idx = torch.empty( Ndim )
#         for n, c in enumerate(inOrder):
#             idx[n] = torch.char.find(outOrder, c)
#         idx = idx.astype('int')

#         # remove undesired dimensions
#         slices = []
#         for i in idx:
#             if i == -1:
#                 slices.append(torch.s_[0])
#             else:
#                 slices.append(torch.s_[:])

#         slices = tuple(slices)
#         data = data[slices]

#         # reorder remaining dimensions
#         order = idx[idx != -1]
#         data = torch.moveaxis(data, torch.arange(Nout), order)

#     return data


def CropEdge(dset, npx=10, edges='l', order: str = 'rzxytc'):
    '''
    It crops an ISM dataset along the specified edges of the xy plane.
    
    Parameters
    ----------
    dset : tensor
        ISM dataset
    npx : int, optional
        Number of pixel to crop from each edge. The default is 10.
    edges : str, optional
        Cropped edges. The possible values are 'l' (left),'r' (right),
        'u' (up), and 'd' (down). Any combination is possible. The default is 'l'.
    order : str, optional
        Order of the dimensions of the dataset The default is 'rzxytc'.

    Returns
    -------
    dset_cropped : tensor
        ISM dataset cropped

    '''

    default_order = 'rzxytc'

    dset_cropped = Reorder(dset, inOrder = order, outOrder = default_order)

    if 'l' in edges:
        dset_cropped = dset_cropped[..., npx:, :, :, :]

    if 'r' in edges:
        dset_cropped = dset_cropped[..., :-npx, :, :, :]

    if 'u' in edges:
        dset_cropped = dset_cropped[..., :, npx:, :, :]

    if 'd' in edges:
        dset_cropped = dset_cropped[..., :, :-npx, :, :]

    return Reorder(dset_cropped, inOrder = default_order, outOrder = order)


def DownSample(dset, ds: int = 2, order: str = 'rzxytc'):
    '''
    It downsamples an ISM dataset on the xy plane.
    
    Parameters
    ----------
    dset : tensor
        ISM dataset.
    ds : int, optional
        Downsampling factor. The default is 2.
    order : str, optional
        Order of the dimensions of the dataset The default is 'rzxytc'.
        
    Returns
    -------
    dset_ds : tensor
        ISM dataset downsampled.

    '''

    default_order = 'rzxytc'

    dset = Reorder(dset, inOrder = order, outOrder = default_order)

    dset_ds = dset[..., ::ds, ::ds, :, :]

    return Reorder(dset_ds, inOrder = default_order, outOrder = order)


def UpSample(dset, us: int = 2, npx: str = 'even', order: str = 'rzxytc'):
    '''
    It upsamples an ISM dataset on the xy plane.

    Parameters
    ----------
    dset : TYPE
        ISM dataset.
    us : int, optional
        Upsampling factor. The default is 2.. The default is 2.
    npx : str, optional
        Parity of the number of pixels on each axis. The default is 'even'.
    order : str, optional
        Order of the dimensions of the dataset The default is 'rzxytc'.

    Returns
    -------
    dset_us : tensor
        ISM dataset upsampled.

    '''

    default_order = 'rzxytc'

    dset = Reorder(dset, inOrder = order, outOrder = default_order)

    sz = dset.shape

    if npx == 'even':
        sz_us = torch.asarray(sz)
        sz_us[2] = sz_us[2] * us
        sz_us[3] = sz_us[3] * us
    elif npx == 'odd':
        sz_us = torch.asarray(sz)
        sz_us[2] = sz_us[2] * us - 1
        sz_us[3] = sz_us[3] * us - 1

    dset_us = torch.zeros(sz_us)
    dset_us[..., ::us, ::us, :, :] = dset

    return Reorder(dset_us, inOrder = default_order, outOrder = order)


def ArgMaxND(data):
    '''
    It finds the the maximum and the corresponding indeces of a N-dimensional array.

    Parameters
    ----------
    data : tensor
        N-dimensional array.

    Returns
    -------
    arg : tensor(int)
        indeces of the maximum.
    mx : float
        maximum value.

    '''

    idx = torch.argmax(data)

    mx = torch.array(data).ravel()[idx]

    arg = torch.unravel_index(idx, torch.array(data).shape)

    return arg, mx


def FWHM(x, y, height  = 0.5):
    '''
    It calculates the Full Width at Half Maximum of a 1D curve.

    Parameters
    ----------
    x : tensor
        Horizontal axis.
    y : tensor
        Curve.

    Returns
    -------
    FWHM: float
        Full Width at Half Maximum of the y curve.

    '''

    height_half_max = torch.max(y) * height
    index_max = torch.argmax(y)
    x_low = torch.interp(height_half_max, y[:index_max], x[:index_max])
    x_high = torch.interp(height_half_max, torch.flip(y[index_max:]), torch.flip(x[index_max:]))
    fwhm = x_high - x_low

    return fwhm, [x_low, x_high]


def RadialSpectrum(img, pxsize: float = 1, normalize: bool = True):
    '''
    It calculates the radial spectrum of a 2D image.

    Parameters
    ----------
    img : tensor
        2D image.
    pxsize : float, optional
        Pixel size. The default is 1.
    normalize : bool, optional
        If True, the result is divided by its maximum. The default is True.

    Returns
    -------
    ftR : tensor
        Radial spectrum.
    space_f : tensor
        Frequency axis.

    '''

    fft_img = torch.fft.fftn(img, axes=[0, 1])
    fft_img = torch.abs(torch.fft.fftshift(fft_img, axes=[0, 1]))

    sx, sy = fft_img.shape
    c = (sx // 2, sy // 2)

    space_f = torch.fft.fftfreq(sx, pxsize)[:c[0]]

    ftR = radial_profile(fft_img, c)

    ftR = ftR[0][:c[0]] / ftR[1][:c[0]]

    ftR = torch.real(ftR)

    if normalize == True:
        ftR /= torch.max(ftR)

    return ftR, space_f


def fingerprint(dset, volumetric=False):
    """
    Calculate the fingerprint of an ISM dataset.
    The last dimension has to be the spad array channel.

    Parameters
    ----------
    dset : torch.array(Nz x Nx x Nx x ... x N*N)
        ISM dataset
    volumetric : bool
        if true, a fingerprint is returned for each axial plane

    Returns
    -------
    Fingerprint : torch.array(Nz x N x N)
        Finger print

    """

    N = int(torch.sqrt(dset.shape[-1]))

    if volumetric == True:
        Nz = dset.shape[0]
        f = torch.empty((Nz, N * N))
        axis = tuple(range(1, dset.ndim - 1))
        f = torch.sum(dset, axis=axis)
        f = f.reshape(Nz, N, N)
    else:
        axis = tuple(range(dset.ndim - 1))
        f = torch.sum(dset, axis=axis)
        f = f.reshape(N, N)
    return f

def point_cloud_from_img(dset):
    """
    Transform the image (or stack of images) into a point cloud matrix.
    The matrix

    Parameters
    ----------
    dset : torch.tensor
        Image (Nz x Ny x Nx)

    Returns
    -------
    point_cloud_matrix : torch.tensor
        Point cloud matrix (Nz*Ny*Nx x 4)

    """
    shape = dset.shape

    N = dset.size

    indices = torch.array(torch.unravel_index(range(N), shape)).T

    values = dset.flatten()

    point_cloud_matrix = torch.column_stack((indices, values))

    return point_cloud_matrix


def kl_divergence(ground_truth, reconstruction, remove_inf=True, intensity_offset = False, normalize_entries = False):
    """
    Calculates the Kullback-Leibler divergence for each iteration of the reconstruction

    Parameters
    ----------
    ground_truth : torch.tensor
        Reference image (Ny x Nx)
    reconstruction : torch.tensor
        Stack of reconstructed images (N_iter x Ny x Nx)
    remove_inf : bool
        If True, local infinity values are replaced with zeros
    intensity_offset :
        If False, the divergence is calculated as the relative entropy.
        If true, it contains an additional term -x + y. 

    Returns
    -------
    kl : torch.tensor
        KL divergence (N_iter)
    """

    if intensity_offset is True:
        from scipy.special import kl_div as kl_div
    else:
        from scipy.special import rel_entr as kl_div
        
    if normalize_entries is True:
        norm_gt = ground_truth.sum()
        norm_data = reconstruction.sum(axis = (-1,-2))

        gt = ground_truth.copy()
        gt = torch.divide(ground_truth, norm_gt, out = gt, where = (norm_gt>0) )

        data = torch.moveaxis(reconstruction, 0, -1).copy()
        data = torch.divide(data, norm_data, out = data, where = (norm_data>0) )
        data = torch.moveaxis(data, -1, 0)
    else:
        gt = ground_truth
        data = reconstruction

    n_iter = reconstruction.shape[0]
    kl = torch.empty(n_iter)

    for n in range(n_iter):
        kl_iter = kl_div(gt, data[n])
        if remove_inf is True:
            kl_iter[torch.isposinf(kl_iter)] = 0
        kl[n] = kl_iter.sum(axis=(-2, -1))

    return kl


def normalized_absolute_difference(ground_truth, reconstruction):
    """
    Calculates the normalized absolute difference between two images

    Parameters
    ----------
    ground_truth : torch.tensor
        Reference image (Ny x Nx)
    reconstruction : torch.tensor
        Reconstructed images (Ny x Nx)

    Returns
    -------
    nad : float
        Normalized absolute difference
    """

    tot_ref = ground_truth.sum()
    tot_img = reconstruction.sum()

    nad = torch.abs(reconstruction / tot_img - ground_truth / tot_ref)

    return nad


def check_saturation(dset, sat_map = None):
    """
    Checks each channel for saturation.

    Parameters
    ----------
    dset : torch.tensor
        Raw ISM dataset. The channel dimension must be the last one.
    sat_map : torch.tensor
        Saturation value for each channel (Nch).
    """

    if sat_map is None:
        sat_map = torch.ones((5, 5)) * 4
        sat_map[1:-1, 1:-1] = 5
        sat_map[2, 1:-1] = 6
        sat_map[1:-1, 2] = 6
        sat_map[2, 2] = 10
        sat_map = 2**sat_map - 1
        sat_map = sat_map.flatten()

    n_ch = dset.shape[-1]
    n_sat = torch.empty(n_ch).astype('int')
    n_tot = torch.size(dset[..., 0])

    print('\nSaturated pixels: \n')

    for c in range(n_ch):
        n_sat[c] = torch.size( dset[..., c][dset[..., c] == sat_map[c]] )
        percent = 100 * n_sat[c] / n_tot
        print(rf'Channel {c:02d}: {n_sat[c]}/{n_tot} ({percent:.2f} %)')


def GaussMultVar(X, Y, M1, M2):
    """
    Multivariate Gaussian function.

    Parameters
    ----------
    X: torch.tensor
        X axis.
    Y : torch.tensor
        Y axis.
    M1: torch.tensor
        First momentum of the distribution (average)
    M2: torch.tensor
        Second momentum of the distribution (variance matrix)

    Returns
    -------
    g : torch.tensor
        Image of the multivariate Gaussian function
    """

    from numpy.linalg import inv

    S = torch.asarray([X, Y])
    S = torch.moveaxis(S, 0, 2) - M1

    A = inv(M2)

    B = torch.einsum('ij, lmj -> ilm', A, S)
    C = torch.einsum('ijk, kij -> ij', S, B)

    g = torch.exp(- 0.5 * C)

    return g


def fit_to_gaussian(img, pxsize, baseline=False, p0 = None):
    """
    Fit an image to a multivariate Gaussian function

    Parameters
    ----------
    img: torch.tensor
        2D image.
    pxsize : float
        Size of the pixe of the image.
    baseline : bool
        If True, the fit model adds to a constant baseline.
    p0: tuple
        Starting parameters for the fitting.
        The first two are the elements of the first moment vector.
        The next three are the elements of the second moment matrix.
        The next one is the amplitude.
        If next one is the baseline value (to be used only is baseline is True).

    Returns
    -------
    img_fit : torch.tensor
        Image of the result of the fit.
    sigma_matrix_diag: torch.tensor
        Square root of the diagonalized variance matrix.
    popt : torch.tensor
        Array of the fitted parameters.
    """

    import scipy.optimize as opt
    from numpy.linalg import eig

    sz = img.shape

    y = pxsize * (torch.arange(sz[0]) - sz[0] // 2)
    x = pxsize * (torch.arange(sz[1]) - sz[1] // 2)

    X, Y = torch.meshgrid(x, y)

    if baseline is False:
        if p0 is None:
            p0 = (0, 0, 1000, 0, 1000, 1)
        fit_model = lambda xdata, a, b, c, d, e, f: f * GaussMultVar(xdata[0].reshape(sz), xdata[1].reshape(sz),
                                                                 torch.asarray([a, b]),
                                                                 torch.asarray([[c, d], [d, e]])).ravel()
    elif baseline is True:
        if p0 is None:
            p0 = (0, 0, 1000, 0, 1000, 1, 0)
        fit_model = lambda xdata, a, b, c, d, e, f, g: g + f * GaussMultVar(xdata[0].reshape(sz), xdata[1].reshape(sz),
                                                                            torch.asarray([a, b]),
                                                                            torch.asarray([[c, d], [d, e]])).ravel()

    xdata = torch.vstack((X.ravel(), Y.ravel()))

    popt, pcov = opt.curve_fit(fit_model, xdata, img.ravel(), p0)

    img_fit = fit_model(xdata, *popt).reshape(sz)

    var_matrix = torch.asarray([[popt[2], popt[3]], [popt[3], popt[4]]])
    var_matrix_diag = torch.diag(eig(var_matrix)[0])
    sigma_matrix_diag = torch.sqrt(var_matrix_diag)

    # D4sigma = 4 * torch.sqrt(sigma_matrix_diag) / 1e3

    return img_fit, sigma_matrix_diag, popt
