import numpy as np
from Tidy_Functions import Polcam, LocalSegment, Unwrap, RedefineCentre, \
    NormalisingWidth, BinAveraging, ThreshThick, FitGaussian, maxcoord_saver, Otsu

def Everything(imagelist, colors = False, aolps = False, dolps = False, 
               s0s = True, masks = False, unwraps0 = True, unwrapaolp = False, 
               unwrapdolp = False, unwrapcolor = False, maxcoordinates = True, 
               xaxis = True, profiles = False, multprofiles = False, 
               fittings = False, FWHMs = False, thickness = 50, 
               binsize = 100, normintensity = False, normalwidth = False, 
               targetwidth = 2000, polariser_string = '[-45 0; 90 45]'):
    '''

    Parameters
    ----------
    imagelist : TYPE
        DESCRIPTION.
    colors : TYPE, optional
        DESCRIPTION. The default is False.
    aolps : TYPE, optional
        DESCRIPTION. The default is False.
    dolps : TYPE, optional
        DESCRIPTION. The default is False.
    s0s : TYPE, optional
        DESCRIPTION. The default is True.
    masks : TYPE, optional
        DESCRIPTION. The default is False.
    unwraps0 : TYPE, optional
        DESCRIPTION. The default is True.
    unwrapaolp : TYPE, optional
        DESCRIPTION. The default is True.
    unwrapdolp : TYPE, optional
        DESCRIPTION. The default is True.
    unwrapcolor : TYPE, optional
        DESCRIPTION. The default is False.
    maxcoordinates : TYPE, optional
        DESCRIPTION. The default is True.
    xaxis : TYPE, optional
        DESCRIPTION. The default is True.
    profiles : TYPE, optional
        DESCRIPTION. The default is False.
    multprofiles : TYPE, optional
        DESCRIPTION. The default is False.
    fittings : TYPE, optional
        DESCRIPTION. The default is False.
    FWHMs : TYPE, optional
        DESCRIPTION. The default is False.
    thickness : TYPE, optional
        DESCRIPTION. The default is 50.
    binsize : TYPE, optional
        DESCRIPTION. The default is 100.
    normintensity : TYPE, optional
        DESCRIPTION. The default is False.
    normalwidth : TYPE, optional
        DESCRIPTION. The default is False.
    targetwidth : TYPE, optional
        DESCRIPTION. The default is 2000.
    polariser_string : TYPE, optional
        DESCRIPTION. The default is '[-45 0; 90 45]'.

    Returns
    -------
    results : TYPE
        DESCRIPTION.
    key_list : TYPE
        DESCRIPTION.

    '''    
    colorlist = []
    aolplist = []
    dolplist = []
    s0list = []
    
    masklist = []
    
    unwrappeds0list = []
    unwrappeddolplist = []
    unwrappedaolplist = []
    unwrappedcolorlist = []
    
    maxcoordlist = []
    
    profiless0list = []
    threshedimages = []
    profilesmultipliedlist = []
    
    xaxeslist = []
    fitdatalist = []
    FWHMlist = []

    if not isinstance(imagelist, list):
        current = [imagelist]
    else:
        current = imagelist
        
    for i in range(len(current)):
        color, aolp, dolp, s0 = Polcam(current[i], 
                                       polariser_string = polariser_string)
        if colors:    
            colorlist.append(color)
        if aolps:
            aolplist.append(aolp)
        if dolps:    
            dolplist.append(dolp)
        if s0s:    
            s0list.append(s0)
        
        mask = LocalSegment(image = s0, erosions = 13, dilations = 0)   
        
        # mask = Otsu(image = s0)
        
        if masks:    
            masklist.append(mask)
        
        if normintensity:    
            unwrappeds0 = Unwrap(s0, mask, thickness = thickness, smoothing = 5, normalising = True)
        else:
            unwrappeds0 = Unwrap(s0, mask, thickness = thickness, smoothing = 5, normalising = False)
        
        unwrappeds0, startx = RedefineCentre(unwrappeds0)  
        
        if normalwidth:  
            unwrappeds0 = NormalisingWidth(unwrappeds0, targetwidth = targetwidth)
        if unwraps0:    
            unwrappeds0list.append(unwrappeds0)
        
        
        if unwrapdolp:
            unwrappeddolp = Unwrap(dolp, mask, thickness = thickness, smoothing = 5, normalising = False)
            unwrappeddolp = RedefineCentre(unwrappeddolp, startx)
            if normalwidth: 
                unwrappeddolp = NormalisingWidth(unwrappeddolp, targetwidth = 2000)
            unwrappeddolplist.append(unwrappeddolp)
            
        if unwrapaolp:
            unwrappedaolp = Unwrap(aolp, mask, thickness = thickness, smoothing = 5, normalising = False)
            unwrappedaolp = RedefineCentre(unwrappedaolp, startx)
            if normalwidth:    
                unwrappedaolp = NormalisingWidth(unwrappedaolp, targetwidth = 2000)
            unwrappedaolplist.append(unwrappedaolp)
        
        if unwrapcolor:
            unwrappedcolor = Unwrap(color, mask, thickness = thickness, smoothing = 5, normalising = False)
            unwrappedcolor = RedefineCentre(unwrappedcolor, startx)
            if normalwidth:    
                unwrappedcolor = NormalisingWidth(unwrappedcolor, targetwidth = 2000)
            unwrappedcolorlist.append(unwrappedcolor)
        
        if maxcoordinates:
            maxcoord = maxcoord_saver(unwrappeds0)
            maxcoordlist.append(maxcoord)
        
        if profiles:    
            profiless0 = BinAveraging(unwrappeds0, bin_size = binsize)
            profiless0list.append(profiless0)
            if multprofiles:
                threshed_unwrapped = ThreshThick(image = unwrappeds0, block = 31, it = 2, offset = 5)
                threshedimages.append(threshed_unwrapped)
                multiplied_image = threshed_unwrapped * unwrappeds0
                profilesmultiplied = BinAveraging(image = multiplied_image, bin_size = binsize)
                profilesmultipliedlist.append(profilesmultiplied)

        if xaxis:
            xax = np.arange(len(unwrappeds0))
            xaxeslist.append(xax)
        
        if fittings:
            xax = np.arange(len(unwrappeds0))        
            fitting = []
            fwhm = []
            for j in range(len(profilesmultiplied)):
                y = profilesmultiplied[j]
                # y = y[len(y)/10 : 9*len(y)/10]          #just fitting over the middle 80% of the curves
                # x = xax[len(xax)/10 : 9*len(xax)/10]
                fitdata, FWHM = FitGaussian(xdata = xax, ydata = y, FWHM = True)
                fitting.append(fitdata)
                fwhm.append(FWHM)
            
            fitdatalist.append(fitting)
            FWHMlist.append(fwhm)
    
    results = {}

    # Only add to the dictionary if the flag was True
    if colors:          results['colors'] = colorlist
    if aolps:           results['aolps'] = aolplist
    if dolps:           results['dolps'] = dolplist
    if s0s:             results['s0s'] = s0list
    if masks:           results['masks'] = masklist
    if unwraps0:        results['unwrapped_s0'] = unwrappeds0list
    if unwrapdolp:      results['unwrapped_dolp'] = unwrappeddolplist
    if unwrapaolp:      results['unwrapped_aolp'] = unwrappedaolplist
    if unwrapcolor:     results['unwrapped_color'] = unwrappedcolorlist
    if maxcoordinates:  results['max_coords'] = maxcoordlist
    if profiles:        results['profiles_s0'] = profiless0list
    if multprofiles:    results['profiles_mult'] = profilesmultipliedlist
    if multprofiles:    results['threshed_image'] = threshedimages
    if xaxis:           results['x_axes'] = xaxeslist
    if fittings:        results['fittings'] = fitdatalist
    if FWHMs:           results['fwhms'] = FWHMlist
    
    key_list = results.keys()
    return results, key_list