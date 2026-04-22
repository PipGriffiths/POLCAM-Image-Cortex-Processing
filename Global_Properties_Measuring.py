from skimage import measure

def maj_min_axes(mask, scale):
    region = measure.regionprops(measure.label(mask))[0]
    major_axis = region.major_axis_length * scale
    minor_axis = region.minor_axis_length * scale
    return major_axis, minor_axis