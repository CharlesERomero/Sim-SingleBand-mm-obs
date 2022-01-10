import numpy as np
import scipy.ndimage
import scipy.signal

def get_freqarr_2d(nx,ny,psx, psy):
    """
       Compute frequency array for 2 D FFT transform
      
       Parameters
       ----------
       nx : integer
            number of samples in the x direction
       ny : integer
            number of samples in the y direction
       psx: integer
            map pixel size in the x direction       
       psy: integer
            map pixel size in the y direction
    
       Returns
       -------
       k : float 2D numpy array
           frequency vector
    """
    kx =  np.outer(np.fft.fftfreq(nx),np.zeros(ny).T+1.0)/psx
    ky =  np.outer(np.zeros(nx).T+1.0,np.fft.fftfreq(ny))/psy
    k = np.sqrt(kx*kx + ky*ky)
    return k

def power_spectrum_2d(arr,nbins=10,psx=1,psy=1,logbins=0):
    """
    Compute 2D power spectrum of arr
    
    Parameters
    ----------
    arr: float 2D numpy array
         2D array for which we compute the power spectrum
    nbins: integer, optional
         number of frequency k bins (10)
         
    psx:  integer, optional (1)
    Returns
    -------
    kbin: float 1D numpy array
          bins in k-space
    pkbin: float 1D numpy array
          2D power spectrum for kbin
    """
    farr = np.fft.fft2(arr)/np.double(arr.size)
    nx,ny = arr.shape
    k = get_freqarr_2d(nx,ny,psx, psy)
    pk = np.double(farr * np.conj(farr))
    if logbins:
        kbinarr = np.logspace(0.0,np.log(k.max()),nbins+1) 
    else:
        kbinarr = np.arange(nbins+1)/np.double(nbins)*(k.max()-k.min())
    kbin   = np.zeros(nbins+1)
    pkbin  = np.zeros(nbins+1)
    pkbins = np.zeros(nbins+1)

    for idx in range(0,nbins):
        list = np.nonzero((k > kbinarr[idx]) * (k<= kbinarr[idx+1]))
        kbin[idx+1] = np.median(k[list])
        pkbin[idx+1] = np.mean(pk[list])
        pkbins[idx+1] = np.std(pk[list])/np.sqrt(len(list[0]))
    return kbin,pkbin,pkbins

def simu_gaussian_noise(nx,ny,pk):
    """
     Simulate a 2D map assuming Gaussian noise as computed from 2D
     power spectrum
     
     
     To be DONE !!!!!
    """
    simumap = np.random.normal(0.0,1.0,[nx,ny])
    karr = get_freqarr_2d(nx,ny,psx, psy)
    
    return
def cross_power_spectrum_2d(arr,arr1,nbins=10,psx=1.0,psy=1.0,logbins=0):
    nx,ny = arr.shape
    nx1,ny1 = arr1.shape
    if nx1 == nx:
        if ny1 == ny:
            farr = np.fft.fft2(arr)/np.double(arr.size)
            farr1 = np.fft.fft2(arr1)/np.double(arr1.size)
            k = get_freqarr_2d(nx,ny,psx, psy)
            pk = np.double(farr * np.conj(farr1))
            if logbins:
                kbinarr = np.logspace(0.0,np.log(k.max()),nbins+1) 
            else:
                kbinarr = np.arange(nbins+1)/np.double(nbins)*(k.max()-k.min())
            kbin = np.zeros(nbins+1)
            pkbin = np.zeros(nbins+1)
            for idx in range(0,nbins):
                list = np.nonzero((k > kbinarr[idx]) * (k<= kbinarr[idx+1]))
                kbin[idx+1] = np.median(k[list])
                pkbin[idx+1] = np.mean(pk[list])
                pkbins[idx+1] = np.std(pk[list])/np.sqrt(len(list[0]))
    return kbin,pkbin,pkbins

def fourier_conv_2d(arr,kernel,doshift=True):
    """
    If kernel (e.g. a Gaussian) is peaked in the center, then you
    want to have doshift=True.
    """
    farr = np.fft.fft2(arr)
    fker = np.fft.fft2(kernel)
    farr = farr * fker
    arrc = np.real(np.fft.ifft2(farr))
    if doshift:
        arrc = np.fft.fftshift(arrc)
    
    return arrc

def fourier_filtering_2d(arr,filt_type,par):
    farr  = np.fft.fft2(arr)
    nx,ny = arr.shape
    kx    = np.outer(np.fft.fftfreq(nx),np.zeros(ny).T+1.0)
    ky    = np.outer(np.zeros(nx).T+1.0,np.fft.fftfreq(ny))
    k     = np.sqrt(kx*kx + ky*ky)
    if filt_type == 'gauss': filter = gauss_filter_2d(k,par)
    if filt_type == 'hpcos': filter = hpcos_filter_2d(k,par)
    if filt_type == 'lpcos': filter = lpcos_filter_2d(k,par)
    if filt_type == 'bpcos': filter = bpcos_filter_2d(k,par)
    if filt_type == 'tab':   filter = table_filter_2d(k,par)
    farr    = farr * filter
    arrfilt = np.real(np.fft.ifft2(farr))
    return arrfilt

def gauss_filter_2d(k,par):
    fwhm = par
    sigma = fwhm/(2.0*np.sqrt(2.0*np.log(2)))
    filter = np.exp(-2.0*k*k*sigma*sigma*np.pi*np.pi)
    return filter

def lpcos_filter_2d(k,par):
    k1 = par[0]
    k2 = par[1]
    filter = k*0.0
    filter[k < k1]  = 1.0
    filter[k >= k1] = 0.5 * (1+np.cos(np.pi*(k[k >= k1]-k1)/(k2-k1)))
    filter[k > k2]  = 0.0
    return filter

def hpcos_filter_2d(k,par):
    k1 = par[0]
    k2 = par[1]
    filter = k*0.0
    filter[k < k1]  = 0.0
    filter[k >= k1] = 0.5 * (1-np.cos(np.pi*(k[k >= k1]-k1)/(k2-k1)))
    filter[k > k2]  = 1.0
    return filter

def bpcos_filter_2d(k,par):
    filter = hpcos_filter_2d(k,par[0:2]) * lpcos_filter_2d(k,par[2:4])
    return filter


def gauss_2d(sigma,nx,ny):
    ix =  np.outer(np.arange(nx),np.zeros(ny).T+1)-nx/2
    iy =  np.outer(np.zeros(nx)+1,np.arange(ny).T)-ny/2
    r = ix*ix+iy*iy
    fg = np.exp(-0.5*r/sigma/sigma)    
    return fg

def table_filter_2d(k,par):
    from scipy import interpolate
    kbin,filterbin = par
    f = interpolate.interp1d(kbin, filterbin)
    kbin_min = kbin.min()
    kbin_max = kbin.max()
    
    filter = k * 0.0
    filter[(k >= kbin_min)  & (k <= kbin_max)] = f(k[(k >= kbin_min)  & (k <= kbin_max)])   # use interpolation function returned by `interp1d`
    filter[(k < kbin_min)] = filterbin[kbin == kbin_min]
    filter[(k > kbin_max)] = filterbin[kbin == kbin_max]

    return filter

def fourier_filtering_1d(arr,filt_type,par,xarr=[]):
    farr = np.fft.fft(arr)
    if len(xarr) == 0:
        nx = len(arr)         # it's strictly one dimensional!
        k  =  np.fft.fftfreq(nx)
    else:
        nx = len(arr)         # it's strictly one dimensional!
        raise Exception
        
        #    if filt_type == 'gauss': filter = gauss_filter_2d(k,par)
#    if filt_type == 'hpcos': filter = hpcos_filter_2d(k,par)
#    if filt_type == 'lpcos': filter = lpcos_filter_2d(k,par)
#    if filt_type == 'bpcos': filter = bpcos_filter_2d(k,par)
    if filt_type == 'tab': filter = table_filter_1d(k,par)
    farr = farr * filter
    arrfilt = np.real(np.fft.ifft(farr))
    return arrfilt

def table_filter_1d(k,par):
    from scipy import interpolate
    kbin,filterbin = par
    f = interpolate.interp1d(kbin, filterbin)
    kbin_min = kbin.min()
    kbin_max = kbin.max()
    
    filter = k * 0.0
    
    filter[(k >= kbin_min)  & (k <= kbin_max)] = f(k[(k >= kbin_min)  & (k <= kbin_max)])   # use interpolation function returned by `interp1d`
    filter[(k < kbin_min)] = filterbin[kbin == kbin_min]
    filter[(k > kbin_max)] = filterbin[kbin == kbin_max]

    return filter


