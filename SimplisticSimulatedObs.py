import numpy as np
from astropy.io import fits as pyfits

def mk_hex_array(ntop,spacing=0.26):
    """ 
    I'm taking spacing to be in arcminutes (for the case of MUSTANG-2)
    """
    
    nrows = 2*ntop - 1
    pixid = np.array([]); idstart=0
    xpix  = np.array([])
    ypix  = np.array([])
    for i in range(ntop):
        ninr = ntop+i # Number of pixels in row
        xpos = np.arange(ninr) - (ninr-1)/2
        ypos = np.ones(ninr)*(ntop-i-1)*np.sqrt(3)/2
        myid = np.arange(idstart,idstart+ninr)
        xpix = np.append(xpix,xpos)
        pixid= np.append(pixid,myid)
        ypix = np.append(ypix,ypos)
        idstart+=ninr
        if i < ntop-1:  # If not the "central" row
            myid = np.arange(idstart,idstart+ninr)
            xpix = np.append(xpix,xpos)  # Same as before
            pixid= np.append(pixid,myid) # Additional IDs
            ypix = np.append(ypix,-ypos) # Flip y value
            idstart+=ninr

    return pixid, xpix*spacing, ypix*spacing

def fourier_filter(t,tod,ffilt,width=0.05):

    n  = len(t)
    dt = t[1]-t[0]
    freqs = np.fft.fftfreq(n,dt)

    ndet,nint = tod.shape
    tfft = np.fft.fft(tod)

    lpf = np.ones(n)
    hpf = np.ones(n)
    if ffilt[1] != 0:
        lpf = lpcos_filter(freqs,[ffilt[1]*(1-width),ffilt[1]*(1+width)])
    if ffilt[0] != 0:
        hpf = hpcos_filter(freqs,[ffilt[0]*(1-width),ffilt[0]*(1+width)])

    filt    = np.outer(np.ones(ndet),hpf*lpf)
        
    filttod = np.real(np.fft.ifft(tfft*filt))

    return filttod

def lpcos_filter(k,par):
    k1 = par[0]
    k2 = par[1]
    filter = k*0.0
    filter[k < k1]  = 1.0
    filter[k >= k1] = 0.5 * (1+np.cos(np.pi*(k[k >= k1]-k1)/(k2-k1)))
    filter[k > k2]  = 0.0
    return filter

def hpcos_filter(k,par):
    k1 = par[0]
    k2 = par[1]
    filter = k*0.0
    filter[k < k1]  = 0.0
    filter[k >= k1] = 0.5 * (1-np.cos(np.pi*(k[k >= k1]-k1)/(k2-k1)))
    filter[k > k2]  = 1.0
    return filter

def rotate_xy(xymap,theta):

    x,y = xymap
    xy  = np.vstack((x.flatten(),y.flatten()))
    rot = np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    xyp = np.matmul(rot,xy)
    xp  = xyp[0,:].reshape(x.shape)
    yp  = xyp[1,:].reshape(y.shape)

    return xp,yp

def get_scan_atm(scan,elev=45.0,pa=None,zopac=0.1,tatm=270.0):

    if pa is None:
        pa = np.random.rand()*360.0

    theta = pa*np.pi/180.0

    newx,newy = rotate_xy((scan["x"],scan["y"]),theta)
    
    myelev = (newy/60.0 + elev)*np.pi/180.0 # in radians now
    myatm  = 1.0 / np.sin(myelev)             # atmospheres = csc = 1/sin(elev)
    
    T_atm  = tatm * (1 - np.exp(-zopac*myatm))
    detmn  = np.mean(T_atm,axis=1)
    for i in range(scan["ndet"]):
        T_atm[i,:] -= detmn[i]
    
    return T_atm

def read_tod_from_fits(fname,hdu=1,branch=None):
    f=pyfits.open(fname)
    raw=f[hdu].data
    #print 'sum of cut elements is ',np.sum(raw['UFNU']<9e5)

    pixid=raw['PIXID']
    dets=np.unique(pixid)
    ndet=len(dets)
    nsamp=len(pixid)/len(dets)
    if True:
        ff=180/np.pi
        xmin=raw['DX'].min()*ff
        xmax=raw['DX'].max()*ff
        ymin=raw['DY'].min()*ff
        ymax=raw['DY'].max()*ff
        print('nsamp and ndet are ',ndet,nsamp,len(pixid),' on ',fname, 'with lims ',xmin,xmax,ymin,ymax)
    else:
        print('nsamp and ndet are ',ndet,nsamp,len(pixid),' on ',fname)
    #print raw.names
    dat={}
    #this bit of odd gymnastics is because a straightforward reshape doesn't seem to leave the data in
    #memory-contiguous order, which causes problems down the road
    #also, float32 is a bit on the edge for pointing, so cast to float64
    dx=raw['DX']
    if not(branch is None):
        bb=branch*np.pi/180.0
        dx[dx>bb]=dx[dx>bb]-2*np.pi
    #dat['dx']=np.zeros([ndet,nsamp],dtype=type(dx[0]))
    ndet=np.int(ndet)
    nsamp=np.int(nsamp)
    dat['dx']=np.zeros([ndet,nsamp],dtype='float64')
    dat['dx'][:]=np.reshape(dx,[ndet,nsamp])[:]
    dy=raw['DY']
    #dat['dy']=np.zeros([ndet,nsamp],dtype=type(dy[0]))
    dat['dy']=np.zeros([ndet,nsamp],dtype='float64')
    dat['dy'][:]=np.reshape(dy,[ndet,nsamp])[:]
    if 'ELEV' in raw.names:
        elev=raw['ELEV']*np.pi/180
        dat['elev']=np.zeros([ndet,nsamp],dtype='float64')
        dat['elev'][:]=np.reshape(elev,[ndet,nsamp])[:]

    tt=np.reshape(raw['TIME'],[ndet,nsamp])
    tt=tt[0,:]
    dt=np.median(np.diff(tt))
    dat['dt']=dt
    pixid=np.reshape(pixid,[ndet,nsamp])
    pixid=pixid[:,0]
    dat['pixid']=pixid
    dat_calib=raw['FNU']
    #print 'shapes are ',raw['FNU'].shape,raw['UFNU'].shape,np.mean(raw['UFNU']>9e5)
    #dat_calib[raw['UFNU']>9e5]=0.0

    #dat['dat_calib']=np.zeros([ndet,nsamp],dtype=type(dat_calib[0]))
    dat['dat_calib']=np.zeros([ndet,nsamp],dtype='float64') #go to double because why not
    dat_calib=np.reshape(dat_calib,[ndet,nsamp])

    dat['dat_calib'][:]=dat_calib[:]
    if np.sum(raw['UFNU']>9e5)>0:
        dat['mask']=np.reshape(raw['UFNU']<9e5,dat['dat_calib'].shape)
        dat['mask_sum']=np.sum(dat['mask'],axis=0)
    #print 'cut frac is now ',np.mean(dat_calib==0)
    #print 'cut frac is now ',np.mean(dat['dat_calib']==0),dat['dat_calib'][0,0]
    dat['fname']=fname
    f.close()
    return dat

def noise_from_M2data(mfile,t,ndet,shuffle=True):
    """
    This will crudely bootstrap noise realizations from real data. There's probably a lot of nuance
    that goes into the noise based on the detectors, readout, scan trajectory, and scan speed. 
    (This ignores all of that).
    """
    
    dat = read_tod_from_fits(mfile)
    ndm2,nt = dat["dat_calib"].shape
    tm2     = nt*dat["dt"]
    tfactor = np.max(t)/tm2    # How many times longer is the intended scan vs. the real scan?
    #times   = np.arange(nt)*dat["dt"]
    nflips  = int(np.ceil(tfactor))
    print("Nflips: ",nflips)
    
    if nflips > 1:
        dat_flipped = np.flip(dat["dat_calib"],axis=1)
        dat_init_stitch = np.hstack((dat["dat_calib"],dat_flipped))
        dat_stitched = np.tile(dat_init_stitch,int(np.ceil(nflips/2)))[:,:nflips*nt]
    else:
        dat_stitched = dat["dat_calib"].copy()

    t_stitched = dat["dt"]*np.arange(nt*nflips)
    dat_interp = np.zeros((ndm2,len(t)))
    for i in range(ndm2):
        dat_interp[i,:]  = np.interp(t,t_stitched,dat_stitched[i,:])
    ### Now we have timestreams that match the time-length of our intended scan

    ### We need to match it to the number of detectors now.

    det_factor = ndet / ndm2
    if det_factor <= 1:
        dat_out = dat_interp[:ndet,:]
    else:
        ndf = int(np.ceil(det_factor))
        dat_out_init = np.tile(dat_interp.T,(1,ndf)).T
        dat_out = dat_out_init[:ndet,:]

    #print(dat_out.shape)
    if shuffle:
        sind = np.arange(ndet)
        np.random.shuffle(sind)
        #print(sind)
        dat_out = dat_out[sind,:]

    return dat_out

def create_noise(t,ndet,norm=1e1,knee=1.0,slope=0,pdn=3e-2,knee2=0.03):

    """ 
    There is room to be much more sophisticated here...
    """
    
    n  = len(t)
    dt = t[1]-t[0]
    freqs = np.fft.fftfreq(n,dt)
    bi    = (freqs == 0)
    gi    = (freqs > 0)
    mnzf  = np.min(freqs[gi])
    freqs[bi] = mnzf

    pwhite = np.ones(freqs.shape) * norm
    if slope == 0:
        ps = pwhite.copy()
    else:
        p0 = norm / (knee**slope)
        p1 = p0 * np.abs(freqs)**slope
        ps = pwhite+p1
        plk2 = (np.abs(freqs) < knee2)
        pred = (np.abs(freqs[plk2])/knee2)**(0.9*slope)
        ps[plk2] /= pred
    ps[bi] *= 1e-3
        
    phase    = np.random.random(size=n) * 2*np.pi
    #import pdb;pdb.set_trace()
    oneoverf = np.real(np.fft.ifft(np.sqrt(ps) * np.exp(1j * phase))) * np.sqrt(2)

    cm_noise     = np.outer(np.ones(ndet),oneoverf)
    per_detector = np.random.randn(ndet,n)*pdn
    noisetods    = cm_noise + per_detector 

    return noisetods

def set_wts(scan,reset=False):

    for i in range(scan["ndet"]):
        mytod = scan["vals"][i,:]
        mywts = scan["wts"][i,:]
        zwts  = (mywts == 0)
        if not reset:
            gi = (mywts > 0)
            myrms = np.std(mytod[gi])
        else:
            myrms = np.std(mytod)
        newwt = np.ones(mywts.shape)/myrms**2 if myrms > 0 else np.zeros(mywts.shape)

        if not reset:
            #import pdb;pdb.set_trace()
            newwt[zwts] = 0.0 # Restore any flagged sections
        scan["wts"][i,:] = newwt
        
def scan_traj(radius,nperiods=22,tstep = 1.0e-2):
    """
    I'm taking radius to be in arcminutes (again, for MUSTANG-2)
    """
    
     # Slight over-sampling, like the real thing
    radpd = 20*(radius/2)**(1.0/3.0)
    tfinal= radpd*nperiods
    t = np.arange(tfinal/tstep)*tstep
    omega = 2.0/radpd
    #print(t)

    x = radius*np.cos(omega*t)*np.sin(np.pi*omega*t)
    y = radius*np.sin(omega*t)*np.sin(np.pi*omega*t)
    
    return x,y,t

def create_scan_template(x,y,t,xpix,ypix,pixid):

    newx = np.outer(np.ones(xpix.shape),x)
    newy = np.outer(np.ones(ypix.shape),y)
    for i,(xp,yp) in enumerate(zip(xpix,ypix)):
        newx[i,:] += xp
        newy[i,:] += yp

    vals = np.zeros(newx.shape)
    wts  = np.ones(newx.shape)

    scan = {"x":newx,"y":newy,"t":t,"vals":vals,"wts":wts,"ndet":pixid.size}

    return scan

def map2tod(img,xymap,scan):

    x,y = xymap             # In arcseconds
    xf  = x.flatten()
    yf  = y.flatten()
    pixsize = x[1,0]-x[0,0]
    nx,ny = x.shape
    
    newtod = np.zeros(scan["x"].shape)
    newwts = np.zeros(scan["x"].shape)

    for i,(scanx,scany) in enumerate(zip(scan["x"],scan["y"])):
        #print(scanx.shape)
        #import pdb;pdb.set_trace()
        xs = np.round((scanx*60 - np.min(x))/pixsize)
        ys = np.round((scany*60 - np.min(y))/pixsize)
        ### Now perform some checks to make sure indices are WITHIN the map:
        bix = (xs < 0)+(xs > nx-1)
        biy = (ys < 0)+(ys > ny-1)
        bi  = bix+biy
        xs[bi] = 0
        ys[bi] = 0
        ### Convert to integers for proper use as indices
        inds = (xs.astype(int),ys.astype(int))
        newtod[i,:] = img[inds] # Get map values
        newtod[i,bi]= 0.0

    #scan["vals"] = newtod

    return newtod

def tod2map_v2(xymap,scan,noise=False,axis=1,roll=False):

    x,y = xymap             # In arcseconds
    xf  = x.flatten()
    yf  = y.flatten()
    pixsize = x[1,0]-x[0,0]
    nx,ny   = x.shape
    sv      = scan["vals"].copy()
    sw      = scan["wts"].copy()
    if noise:
        sv = np.flip(sv,axis=axis)
        sw = np.flip(sw,axis=axis)
    if roll:
        ndet,nint = sv.shape
        for i in range(ndet):
            shift = int(np.round((np.random.random()-0.5)*nint))
            sv[i,:] = np.roll(sv[i,:],shift)
            sw[i,:] = np.roll(sw[i,:],shift)
    xs = np.round((scan["x"]*60 - np.min(x))/pixsize)
    ys = np.round((scan["y"]*60 - np.min(y))/pixsize)
    xsf   = xs.flatten()
    ysf   = ys.flatten()
    svf   = sv.flatten()
    swf   = sw.flatten()
    svf   = svf*swf     # Weight the data!

    # Perform checks that xs,ys are "in bounds"
    bix = (xsf < 0)+(xsf > nx-1)
    biy = (ysf < 0)+(ysf > ny-1)
    bi  = bix+biy

    nbin = nx*ny
    oned = (xsf*ny + ysf)
    oned[bi] = nbin+2
    #oned = (xs*ny + ys).astype(int)
    bins = np.arange(nbin+1)-0.1
    
    mval,medge = np.histogram(oned,bins=bins,weights=svf)
    mwts,wedge = np.histogram(oned,bins=bins,weights=swf)
    
    mymap = mval.reshape((nx,ny))
    mywts = mwts.reshape((nx,ny))

    gi = (mywts > 0)
    gmap  = mymap.copy()
    gmap[gi] = mymap[gi] / mywts[gi]
    
    return gmap,mywts
    
def tod2map(xymap,scan):

    x,y = xymap             # In arcseconds
    xf  = x.flatten()
    yf  = y.flatten()
    pixsize=x[1,0]-x[0,0]
    xs = np.round(scan["x"]*60/pixsize)*pixsize
    ys = np.round(scan["y"]*60/pixsize)*pixsize
    img = np.zeros(yf.shape)
    wts = np.zeros(yf.shape)
    nx,ny = x.shape
    for i,(xp,yp) in enumerate(zip(xf,yf)):
        if i % nx == 0: print(i/nx)
        c1 = (xs == xp)
        c2 = (ys == yp)
        gi = c1*c2
        if np.sum(gi) > 0:
            data    = scan["vals"][gi]
            weights = scan["wts"][gi]
            #nzero   = (weights > 0)
            #if np.sum(nzero) > 16:
            #    gd = data[nzero]
            gd = np.sum(data)
            gw = np.sum(weights)
        else:
            gd = 0
            gw = 0
        if gw > 0:
            gd /= gw
        else:
            gd = 0
        img[i] = gd
        wts[i] = gw
    img = img.reshape(x.shape)
    wts = wts.reshape(x.shape)

    mymap = {"img":img,"wts":wts}

    return mymap
            
def make_xymap(xsize,ysize,pixsize):

    xpix = int(np.round((xsize*60)/pixsize))
    ypix = int(np.round((ysize*60)/pixsize))
    x1   = (np.arange(xpix)-xpix//2)*pixsize
    y1   = (np.arange(ypix)-ypix//2)*pixsize
    x    = np.outer(x1,np.ones(ypix))
    y    = np.outer(np.ones(xpix),y1)
    
    return x,y
    
def make_2dgauss(x,y,x0,y0,smaj,smin,theta,amp):

    yt  = (y-y0)
    xt  = (x-x0)
    xp,yp = rotate_xy((xt,yt),theta)
    #yt  = (y-y0).flatten()
    #xt  = (x-x0).flatten()
    #xy  = np.vstack((xt,yt))
    #rot = np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    #xyp = np.matmul(rot,xy)
    #xp  = xyp[0,:].reshape(x.shape)
    #yp  = xyp[1,:].reshape(y.shape)

    g1    = np.exp(-xp**2 / (2*smaj**2))
    g2    = np.exp(-yp**2 / (2*smin**2))
    gauss = amp*g1*g2 

    return gauss

def cmsub(tod,wts,usemean=True,fitperdet=False):

    if usemean:
        cm = np.mean(tod,axis=0)
    else:
        cm = np.median(tod,axis=0)
        
    newtod = tod.copy()

    if fitperdet:
        for i,(mytod,mywt) in enumerate(zip(tod,wts)):
            ata = cm**2 * mywt
            atd = cm*mywt*mytod
            amp = atd/ata
            newtod[i,:] = tod[i,:] - cm*amp
    else:
        ntod,nints = tod.shape
        newtod = tod - np.outer(np.ones(ntod),cm)
            
    return newtod
        
