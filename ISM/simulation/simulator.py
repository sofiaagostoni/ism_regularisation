import numpy as np
from scipy.signal import convolve
import copy as cp
import torch
import torch.nn.functional as F
import deepinv as dinv


class ImageSimulator:
    """
    Object with methods to generate the forward model of a (multichannel) microscope.
    The number of dimensions of phantom and the psf should differ by 1 at most.
    In this case, the last dimension of the psf is interpreted as the channel.

    Attributes
    ----------
    image : torch.tensor
        image convolved with the psf and corrupted by shot noise
    clean_image : torch.tensor
        image convolved with the psf without noise
    phantom : torch.tensor
        stucture of the specimen
    psf : torch.tensor
        point spread function. The last dimension is the channel.
    signal : float
        brightness of the sample (units: photon counts)

    Methods
    -------
    blur :
        Generates the clean image.
    poisson_noise :
        Corrupts the clean image with shot noise.
    forward :
        Generates the blurred and noisy image.

    """
    def __init__(self, phantom=None, psf=None, signal=1, z_projection=False):
        self.image = None
        self.clean_image = None
        self.adjoint_image = None
        self.phantom = phantom
        self.psf = psf
        self.signal = torch.tensor(signal)
        self.z_projection = z_projection
        self.clean_transpose = None

    def blur(self):
            gt = self.ground_truth  # [B, C, H, W]
            num_ch = self.psf.ndim - self.phantom.ndim + 2  # +2 for batch and channel
            sz = self.psf.shape  # [B, C, H, W]
            shape = [1, gt.shape[-2], gt.shape[-1], sz[-1]]
            self.clean_image = torch.zeros(shape, dtype=torch.float32)

            # Multi-channel PSF 
            if num_ch == 1:  

                # Case with z projection
                if gt.ndim == 4 and self.z_projection:
                    # loop on focus planes
                    for z in range(sz[0]):
                        # loop on channels
                        for c in range(sz[-1]):
                            # choose current PSF
                            kernel = self.psf[z, :, :, c].unsqueeze(0).unsqueeze(0)
                            conv = F.conv2d(gt, kernel, padding='same')
                            self.clean_image[z, :, :, c] += conv.squeeze(1)

                # case with no z projection
                else:
                    # loop on channel
                    for c in range(sz[-1]):
                        # choose current PSF
                        kernel = self.psf[ :, :, c].unsqueeze(0).unsqueeze(0)
                        conv = F.conv2d(gt, kernel, padding='same')
                        print(conv.shape)
                        self.clean_image[:, :, :, c] = conv

            # PSF and image have the same dimension
            elif num_ch == 0:  # Single-channel PSF

                # Case with z projection
                if gt.ndim == 4 and self.z_projection:

                    # loop on projections
                    for z in range(sz[0]):
                        kernel = self.psf[z].unsqueeze(0).unsqueeze(0)
                        tmp = F.conv2d(gt, kernel, padding='same')
                        self.clean_image[z, :, :, :]  = tmp.sum(dim=1, keepdim=True)

                # Case with no z projection
                else:
                    kernel = self.psf.unsqueeze(0).unsqueeze(0)
                    self.clean_image = F.conv2d(gt, kernel, padding='same')

            else:
                raise Exception("The PSF has fewer dimensions than the phantom.")

            self.clean_image = torch.clamp(self.clean_image, min=0)



    def blur(self):
        gt = self.ground_truth  # [B, C, H, W]
        # num_ch = self.psf.ndim - self.phantom.ndim + 2  # +2 for batch and channel
        sz = self.psf.shape  # [B, C, H, W]

        # filter_0 = dinv.physics.blur.gaussian_blur(sigma_blur, angle=0.0)
        physics_blurr = dinv.physics.Blur(self.psf, padding = 'circular', device=device)
        self.clean_image = physics_blurr(image)
        shape = [1, gt.shape[-2], gt.shape[-1], sz[-1]]
        self.clean_image = torch.zeros(shape, dtype=torch.float32)

        # Multi-channel PSF 
        if num_ch == 1:  

            # Case with z projection
            if gt.ndim == 4 and self.z_projection:
                # loop on focus planes
                for z in range(sz[0]):
                    # loop on channels
                    for c in range(sz[-1]):
                        # choose current PSF
                        kernel = self.psf[z, :, :, c].unsqueeze(0).unsqueeze(0)
                        conv = F.conv2d(gt, kernel, padding='same')
                        self.clean_image[z, :, :, c] += conv.squeeze(1)

            # case with no z projection
            else:
                # loop on channel
                for c in range(sz[-1]):
                    # choose current PSF
                    kernel = self.psf[ :, :, c].unsqueeze(0).unsqueeze(0)
                    conv = F.conv2d(gt, kernel, padding='same')
                    print(conv.shape)
                    self.clean_image[:, :, :, c] = conv

        # PSF and image have the same dimension
        elif num_ch == 0:  # Single-channel PSF

            # Case with z projection
            if gt.ndim == 4 and self.z_projection:

                # loop on projections
                for z in range(sz[0]):
                    kernel = self.psf[z].unsqueeze(0).unsqueeze(0)
                    tmp = F.conv2d(gt, kernel, padding='same')
                    self.clean_image[z, :, :, :]  = tmp.sum(dim=1, keepdim=True)

            # Case with no z projection
            else:
                kernel = self.psf.unsqueeze(0).unsqueeze(0)
                self.clean_image = F.conv2d(gt, kernel, padding='same')

        else:
            raise Exception("The PSF has fewer dimensions than the phantom.")

        self.clean_image = torch.clamp(self.clean_image, min=0)


    def blur_transpose(self):
        gt = self.ground_truth  # [B, C, H, W] or [B, Z, H, W] 
        num_ch = self.psf.ndim - self.phantom.ndim + 2  # +2 for batch and channel
        sz = self.psf.shape
        bias = None

        shape = [1, gt.shape[-2], gt.shape[-1], sz[-1]]
        self.clean_transpose = torch.zeros(shape, dtype=torch.float32)

        # Multi-channel PSF
        if num_ch == 1:

            # Case with z projection
            if gt.ndim == 4 and self.z_projection:

                for z in range(sz[0]):
                    for c in range(sz[-1]):
                        kernel = self.psf[z, :, :, c].unsqueeze(0).unsqueeze(0)
                        kH, kW = kernel.shape[-2:]
                        padding = (kH // 2, kW // 2)
                        deconv = F.conv_transpose2d(gt, kernel,bias, padding)
                        self.clean_transpose[z, :, :, c] += deconv.squeeze(1)

            # Case without z projection
            else:
                for c in range(sz[-1]):
                    kernel = self.psf[ :, :, c].unsqueeze(0).unsqueeze(0)
                    kH, kW = kernel.shape[-2:]
                    padding = (kH // 2, kW // 2)
                    deconv = F.conv_transpose2d(gt, kernel, bias, padding)
                    self.clean_transpose[:, :, :, c] += deconv

        # Single-channel PSF
        elif num_ch == 0: 

            # Case with z projection
            if gt.ndim == 4 and self.z_projection:

                for z in range(sz[0]):
                    kernel = self.psf[z].unsqueeze(0).unsqueeze(0)
                    kH, kW = kernel.shape[-2:]
                    padding = (kH // 2, kW // 2)
                    deconv = F.conv_transpose2d(gt, kernel, bias, padding)
                    self.clean_transpose[z, :, :, :]  = deconv.sum(dim=1, keepdim=True)
            else:
                kernel = self.psf.unsqueeze(0).unsqueeze(0)
                kH, kW = kernel.shape[-2:]
                padding = (kH // 2, kW // 2)
                self.reconstructed_image = F.conv_transpose2d(gt, kernel, bias, padding)

        else:
            raise Exception("The PSF has fewer dimensions than the phantom.")

        self.reconstructed_image = torch.clamp(self.reconstructed_image, min=0)


    def blur_transpose2(self):
        gt = self.ground_truth  # [B, C, H, W] or [B, Z, H, W]
        num_ch = self.psf.ndim - self.phantom.ndim + 2
        sz = self.psf.shape

        B = self.phantom.shape[0]
        self.adjoint_image = torch.zeros_like(self.phantom)

        if num_ch == 1:
            C = sz[-1]
            if self.phantom.ndim == 4 and self.z_projection:
                Z, H, W = self.phantom.shape[1:]
                tmp = torch.zeros((B, Z, H, W), dtype=torch.float32)
                for z in range(sz[0]):
                    for c in range(C):
                        kernel = torch.flip(self.psf[z, ..., c], dims=[-2, -1]).unsqueeze(0).unsqueeze(0)
                        img = blurred[:, c].unsqueeze(1)
                        tmp[:, z] += F.conv2d(img, kernel, padding='same').squeeze(1)
                self.adjoint_image = tmp.sum(dim=1, keepdim=True)
            else:
                for c in range(C):
                    kernel = torch.flip(self.psf[..., c], dims=[-2, -1]).unsqueeze(0).unsqueeze(0)
                    img = blurred[:, c].unsqueeze(1)
                    self.adjoint_image += F.conv2d(img, kernel, padding='same')
        elif num_ch == 0:
            if self.phantom.ndim == 4 and self.z_projection:
                Z, H, W = self.phantom.shape[1:]
                tmp = torch.zeros((B, Z, H, W), dtype=torch.float32)
                for z in range(sz[0]):
                    kernel = torch.flip(self.psf[z], dims=[-2, -1]).unsqueeze(0).unsqueeze(0)
                    img = blurred
                    tmp[:, z] = F.conv2d(img, kernel, padding='same').squeeze(1)
                self.adjoint_image = tmp.sum(dim=1, keepdim=True)
            else:
                kernel = torch.flip(self.psf, dims=[-2, -1]).unsqueeze(0).unsqueeze(0)
                self.adjoint_image = F.conv2d(blurred, kernel, padding='same')
        else:
            raise Exception("The PSF has fewer dimensions than the phantom.")

        self.adjoint_image = torch.clamp(self.adjoint_image, min=0)


    def poisson_noise(self):
        self.image = torch.poisson(self.clean_image)

    def forward(self):
        self.blur()
        self.poisson_noise()

    def forward_blur(self):
        self.blur()
        self.image = self.clean_image

        
    def forward_blur_adjoint(self):
        self.blur()
        self.blur_transpose()
        self.image = self.adjoint_image

    @property
    def ground_truth(self):
        if self.signal.ndim == 1:
            return torch.einsum('i..., i -> i...', self.phantom, self.signal)
        elif self.signal.ndim == 0:
            return self.phantom * self.signal
        else:
            raise Exception("The signal should be a scalar or a 1D array.")

    def copy(self):
        return cp.copy(self)
