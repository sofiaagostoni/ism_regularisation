import torch


def disk(Nx, radius, offset = [0,0]):
    " Create disk shape"
    
    x = torch.arange(-Nx//2, Nx//2, 1) + offset[0]
    y = torch.arange(-Nx//2, Nx//2, 1) + offset[1]
    
    X, Y = torch.meshgrid(x,y)
    
    r = torch.sqrt(X**2 + Y**2)
    
    disko = torch.where(r<=radius, 1, 0)
    
    return disko


def disks_rad(Nx, radius):
    " Create image of 4 disks. "
    space = 40


    offset = (0,space)
    offset1f = (0,-space)
    offset2f = (space,0)
    offset3f = (-space,0)
    offset4f = (0,0)

    disk_f0  = disk(Nx, radius, offset)
    disk_f1  = disk(Nx, radius, offset1f)
    disk_f2  = disk(Nx, radius, offset3f)
    disk_f3  = disk(Nx, radius, offset2f)
    disk_f4  = disk(Nx, radius, offset4f)

    disk_f = disk_f0 + disk_f1 + disk_f2 + disk_f3  + disk_f4
    disk_f = disk_f.unsqueeze(0).unsqueeze(0)
    return disk_f
    

def disks_rad_back(Nx, radius):
    " Create stack of images of 4 disks focus and background. "
    disks = torch.empty(1,2,Nx,Nx)
    space = 40
    offset = (0,space)
    offset1f = (0,-space)
    offset2f = (space,0)
    offset3f = (-space,0)
    offset4f = (0,0)

    disk_f0  = disk(Nx, radius, offset)
    disk_f1  = disk(Nx, radius, offset1f)
    disk_f2  = disk(Nx, radius, offset3f)
    disk_f3  = disk(Nx, radius, offset2f)
    disk_f4  = disk(Nx, radius, offset4f)

    disk_f = disk_f0 + disk_f1 + disk_f2 + disk_f3  + disk_f4

    offset2 = (space,space)
    offset3 = (-space,space)
    offset4 = (+space,-space)
    offset5 = (-space,-space)

    disk_bkg1 = disk(Nx,radius, offset2)
    disk_bkg2 = disk(Nx,radius, offset3)
    disk_bkg3 = disk(Nx,radius, offset4)
    disk_bkg4 = disk(Nx,radius, offset5)

    disk_bkg = disk_bkg1 + disk_bkg2 + disk_bkg3+ disk_bkg4 
    
    disks = torch.stack((disk_f, disk_bkg), axis=0)
    disks = disks.unsqueeze(0)

    return disks