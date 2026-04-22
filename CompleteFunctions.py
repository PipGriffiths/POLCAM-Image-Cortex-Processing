#Complete Functions 

def TheWholeLotTelophase(image, block, thickness, polariser_string, normalising):
    
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    from Tidy_Functions import Polcam, Segment_Auto_Local, follow_path, Unwrap, BinAveraging, \
        counter, UnwrappedThresholding, CortexSim, ThreshThick
    
    def ProfileSpecBin(image, values):
        profiles = []
        for i in range(len(values)):
            start_x, end_x = values[i]
            current_bin = image[:, start_x : end_x]
            avg_profile = np.mean(current_bin, axis = 1)
            avg_profile = np.flip(avg_profile)
            profiles.append(avg_profile)
        return profiles
    
    colors, aolp, dolp, s0 = Polcam(image, polariser_string = polariser_string)
    dilation, mask = Segment_Auto_Local(s0)
    unwrapped = Unwrap(s0, mask, thickness, 5, normalising = normalising)
    thresh_unwrapped = ThreshThick(unwrapped, block = 31, it = 2)
    multiplied_image = unwrapped * thresh_unwrapped
    maxvalues = np.max(unwrapped, axis = 0)
    heavysmoothing = gaussian_filter(maxvalues, sigma = 80)
    peaks, _ = find_peaks(heavysmoothing)
    
    furrowrange = []
    for i in range(len(peaks)):
        furrowrange.append([peaks[i]- 100, peaks[i] + 100])
    furrowprofiles = ProfileSpecBin(multiplied_image, furrowrange)
    xaxis = [i for i in range(thickness)]
    avgfurrow = np.mean(np.array(furrowprofiles), axis = 0)
    
    def gauss(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    popt,pcov = curve_fit(gauss, xaxis, avgfurrow, p0=[1,20,5])
    FWHM = 2*np.sqrt(2*np.log(2))*popt[2]

    fitdata = gauss(xaxis, *popt)
    
    return xaxis, fitdata, avgfurrow, furrowprofiles, FWHM