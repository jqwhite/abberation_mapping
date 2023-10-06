import os
import numpy as np
from numpy import genfromtxt
import shutil
import tifffile
import glob
import json
import argparse
from scipy.spatial.distance import cdist
from collections import OrderedDict
import sys
# sys.path.append('/home/dsaha/.local/lib/python3.7/site-packages/')
import matplotlib.pyplot as plt
from pathlib import Path
import socket
from tqdm import tqdm
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter
from skimage.measure import label as lab
import re
from skimage.transform import downscale_local_mean


def zola_processed(filepath, lam, max_amp=None):
    '''
    processes the output file  of Zola
    filepath = String, entire path to a json file 
    lam = float, wavelength
    max_amp = float, maximum amplitude allowed, default None then all values are allowed

    returns a dictionary of ansi ordered amplitude values and a flag if all the values are below maximum aplitude
    '''
#     normalization = [1, 2, 2, np.sqrt(6), np.sqrt(3), np.sqrt(6),np.sqrt(8),np.sqrt(8),np.sqrt(8),np.sqrt(8), np.sqrt(10), np.sqrt(10), np.sqrt(5), np.sqrt(10), np.sqrt(10)]
    normalization = [1, 2, 2, np.sqrt(6), np.sqrt(3), np.sqrt(6),np.sqrt(8),np.sqrt(8),np.sqrt(8),np.sqrt(8), np.sqrt(10), np.sqrt(10), np.sqrt(5), np.sqrt(10), np.sqrt(10), np.sqrt(12), np.sqrt(12), np.sqrt(12), np.sqrt(12), np.sqrt(12), np.sqrt(12), np.sqrt(14), np.sqrt(14), np.sqrt(14),  np.sqrt(7), np.sqrt(14), np.sqrt(14), np.sqrt(14),]
    # for reference : https://wp.optics.arizona.edu/visualopticslab/wp-content/uploads/sites/52/2016/08/Zernike-Notes-15Jan2016.pdf
    with open(filepath) as json_file:
        try:
            data = np.array(json.load(json_file)['zernike'])


        except:
            print("WARNING : NOT GOOD FILE")
            return dict({'3':-999}), 0

        data = data*lam/2/np.pi # converting to microns
        normalization = normalization[:data.shape[0]] # cropping the normalizion list
        data = [float(d/n) for d,n in zip(data,normalization)] # normalizing
        zerns_ansi = np.arange(len(normalization)).astype('str') # arranging zernike mode num
        
        if max_amp is not None:
            max_amp = [max_amp,]*len(normalization) # creating a list with maximum aplitude values
            if any([np.abs(d)>np.abs(m) for d,m in zip(data, max_amp)]): # check if any value is greater than max amp
                return dict(zip(zerns_ansi,data)), 0
            else:
                return dict(zip(zerns_ansi,data)), 1
        else:
            return dict(zip(zerns_ansi,data)), 1

def setup_param_file(img_path, load_mode=False):
    '''
    setup the param.json file if already not there
    img_path = String, path to the image, the params.json will be created here

    returns a dictionary of default params or exisitng params
    '''
    params_file = f"{os.path.dirname(img_path)}/params.json"
    
    if Path(params_file).is_file(): # if param file already exist load the params
        with open(params_file) as json_file:
            old_params = json.load(json_file)
    elif load_mode:
        print("Params file does not exist")
        return
    else: # else create a param file with default values
        old_params = OrderedDict()
    list_of_params ={'crop_shape':(16,16,16), 'unit':(0.1,0.113,0.113), 'abb_list': [3,5,6,7,8,9,10,11,12,13,14], 'FIJI_MACRO_PATH': "/projects/project-dsaha/insitu_psf/zola_macro_cluster_1p4.ijm", 'COMPUTERNAME' : socket.gethostname(),
    'thresh_bck':99, 'thresh_dist':None, 'save_dir':os.path.dirname(img_path), 'num_folders':50, 'lam':0.515}
    for key, val in list_of_params.items(): # add the keys 
        if key not in old_params.keys():
            old_params[key] = val
    return old_params

# def write_to_param_file(img_path, params):
def write_to_param_file(parameter_filepath, params):
    '''
    write the used params to the params.json file
    img_path = String, params.json will be searched in the parent dir of img_path
    params = dict, parameter dictionary
    '''
    # params_file = f"{os.path.dirname(img_path)}/params.json"
    with open(parameter_filepath, 'w') as fp:
        json.dump(params, fp)

def show_patch(pat):
    '''
    show a 3D patch as maximum projection
    pat = numpy array, 3d array
    '''
    fig, axs = plt.subplots(1,2,figsize=(7,3))
    axs[0].imshow(np.max(pat,0), cmap='magma')
    axs[1].imshow(np.max(pat,1), cmap='magma')
    [aa.axis('OFF') for aa in axs]
    plt.show()
    return axs

def get_patches(pts3d, img3d, patch_shape, headless=False):
    ''' 
    finds 3D patches cropped around given 3D points
    pts3d = numpy array with shape (n,3) for n points
    img3d = numpy array of the image contaitng the points
    patch_shape = 3d tuple of ints as (z,y,x). z axis is only for reference
    headless = Boolean, displays output if False

    returns a numpy array of new points and patches corresponding to the points 
    '''

    _, n, o = patch_shape   # patch dimensions
    img_dims = img3d.shape  #image dimensions
    def f(p):
        a, b, c = p # z, y, x coordinates of points
        # slice a patch around p with dimensions [:, n, o]
        ss = slice(0, img_dims[0]), slice(max(b - n,0), min(b + n,img_dims[1])), slice(max(c - o,0), min(c + o,img_dims[2]))
        patch = img3d[ss]
        # if the patch extracted doesn't satisfy the specified dimensions reutrn None
        if patch.shape != (img_dims[0], n * 2, o * 2): 
            return None
        return patch

    patches = []
    del_indices = []

    # loop over all points
    for i,p in enumerate(pts3d):
        patch = f(p) # extract a patch
        if patch is None: 
            del_indices.append(i)
            continue
        patches.append(patch)
    # Deletes the patches that are at the boundary and could not be cropped because of the given xy patch size
    if not headless:
        print(f"Deleting {len(del_indices)} because patches could not be cropped")
    patches = np.array(patches)
    new_pts3d = np.delete(pts3d, del_indices, axis=0) # delete the points where patches could not be cropped
    return new_pts3d, patches

def bbox_3D(label):
    '''
    3d bounding box around a label 
    img = labelled image

    returns start and stop of all three dimensions

    adapted from "https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array"
    '''
    z = np.any(label, axis=(1, 2))
    y = np.any(label, axis=(0, 2))
    x = np.any(label, axis=(0, 1))
    try:
        zmin, zmax = np.where(z)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]
        xmin, xmax = np.where(x)[0][[0, -1]]
    except:
        return (0,)*6

    return zmin, zmax, ymin, ymax, xmin, xmax

def crop_around_punkta(pat, z_pos, thresh_bck, show=False, xy_add=10, min_z=8, max_z=100, min_xy=2, max_xy=80, headless=False):
    '''
    does a maximum filter of the patch followed by thresholding with thresh, then crops the patches with given number of z planes around z_pos,
    thereafter finds if the patch is a good one for processing
    pat = 3d numpy array, patch
    z_pos = int, the z position within this patch to crop around
    z_add = number of planes to add before and after the crop
    min_z = int, minimum number of planes
    maz_z = int, maximum number of planes
    
    returns the copped patch and a flag
    '''

    zz, yy, xx = pat.shape

    # blur, threshold and label the points in the patch
    blur = gaussian_filter(pat, (1,1,1))
    binary = blur>(blur[z_pos, yy//2, xx//2]-(blur[z_pos, yy//2, xx//2]-np.percentile(blur,3))*0.25)
    label = lab(binary, connectivity=1)

    # keep only the label that belongs to the punkta at z_pos
    masked_label = label.copy()
    masked_label[masked_label!=masked_label[z_pos,yy//2, xx//2]] = 0

    # crop a bounding box around that label with added number of planes
    zstart_1, zstop_1, ystart_1, ystop_1, xstart_1, xstop_1 = bbox_3D(masked_label)
    
    z_patches = OrderedDict()

    ystart = ystart_1-xy_add
    ystop = ystop_1+xy_add
    xstart = xstart_1-xy_add
    xstop = xstop_1+xy_add

    for z_add in range(14,9,-2): 

        zstart = zstart_1-z_add
        zstop = zstop_1+z_add

        # Deleting because of label touching bounidng box
        if any([ystart<=0, ystop>=yy, xstart<=0, xstop>=xx, zstart<=0, zstop>=zz]):
            if not headless:
                print("Deleting because of label touching bounidng box")
            if show:
                print(f"Cant show because z_start : {zstart_1}, z_stop : {zstop_1}, y_start : {ystart_1}, y_stop : {ystop_1}, x_start : {xstart_1}, x_stop : {xstop_1}")
                show_patch(pat)
            return None
        # Deleting because the patch is either too small or too big
        elif any([zstop-zstart < min_z, ystop-ystart < min_xy, xstop-xstart < min_xy]):
            if not headless:
                print("Deleting because the patch is too small")
            if show:
                show_patch(pat[zstart:zstop, ystart:ystop, xstart:xstop])
            return None
        # Deleting because the patch is too big
        elif any([zstop-zstart>max_z, ystop-ystart>max_xy, xstop-xstart>max_xy]):
            if not headless:
                print("Deleting because the patch is too big")
            if show:
                show_patch(pat[zstart:zstop, ystart:ystop, xstart:xstop])
            return None
        # Deleting because of snr
        z_patch = pat[zstart:zstop, ystart:ystop, xstart:xstop]
        if (np.max(z_patch) - np.median(z_patch))<thresh_bck: 
            if not headless:
                print(f"Deleting becuase of snr {np.max(z_patch) - np.median(z_patch)}")
            if show:
                show_patch(pat[zstart:zstop, ystart:ystop, xstart:xstop])
            return None
        # Deleting because of multiple labels in the cropbox
        if len(np.unique(label[zstart:zstop, ystart:ystop, xstart:xstop]))!=2: # background and one punkta
            if not headless:
                print("Deleting because of multiple labels in the cropbox")
            if show:
                show_patch(pat[zstart:zstop, ystart:ystop, xstart:xstop])
            return None
        z_patches[z_add] = z_patch
    return z_patches.copy()

# def find_psf_crops(img_path, crop_shape=(16,)*3, num_folders=50, thresh_bck=100, threhold_dist=10, show_ignored=False, show_accepted=False, headless=False):
#     '''
#     find, select and save patches
#     img_path = String, path to the image
#     crop_shape = 3d tuple of ints as (z,y,x). z axis is only for reference
#     num_folders = int, for parallel processing, images will be saved in as many num_folders
#     thresh_bck = float, absolute cut off value for SNR, depends on the images
#     show_accepted, show_ignored = Boolean, flags for visualization of accepted or ignored patches
#     headless = Boolean, displays output if False

#     '''
#     dirpath = os.path.dirname(img_path)
#     img = tifffile.imread(img_path)

#     if headless:
#         show_ignored = False
#         show_accepted = False

#     # load points found in Fiji
#     results_csv_path = dirpath+"/Results.csv"
#     try:
#         fiji_points = genfromtxt(results_csv_path, delimiter=',').astype(int) #load the 2d points from scv file
#     except:
#         print("Cannot load Results.csv file")
#     if fiji_points.shape[-1]==3:
#         points2d = fiji_points[1:,1:] # leave out the first row (contains strings X and Y ) and first column (contains enumeration of points)
#     else:
#         points2d = fiji_points[1:,:]  # leave out the first row (contains strings X and Y ) 
#     points2d[:,[1,0]] = points2d[:,[0,1]] # flip x and y co-ordinates
#     if not headless:
#         print(f"Found {points2d.shape[0]} points")

#     # throw out points that are very close determined by threhold_dist
#     # threhold_dist = np.maximum(8,crop_shape[-1]/3)
#     # threhold_dist = 10
#     dist = (np.sort(cdist(points2d, points2d),1)[:,1:]<threhold_dist)[:,0]
#     if not headless:
#         print(f"Deleting {sum(dist==True)} because points are too close")
#     points2d = np.delete(points2d, np.where(dist==True),0)

#     # find the location of maximum pixel along z axis and create 3d points
#     zmax = [np.argmax(img[:,_p[0],_p[1]]) for _p in points2d]
#     points = np.insert(points2d, 0, zmax, 1)

#     # crop patches around the 3d points
#     new_points, patches = get_patches(points.copy(), img, patch_shape=tuple([int(t/2) for t in crop_shape]), headless=headless);

#     processed_counter = 0
#     accepted_counter = 0
#     rejected_counter = 0

#     # remove all the directories and make new ones
#     patches_dirs = glob.glob(f"{dirpath}/patches_for_zola/patches_*/")
#     [shutil.rmtree(p, ignore_errors=True) for p in patches_dirs]
#     [os.makedirs(f"{dirpath}/patches_for_zola/patches_{i}/") for i in range(num_folders)]  
        
#     # loop over the points and patches
#     for pt, patch in tqdm(zip(new_points, patches), disable=headless): 

#         i = int(np.random.uniform(low = 0, high=num_folders))
#         patches_dir = f"{dirpath}/patches_for_zola/patches_{i}/"

#         # crop th patch around the punkta
#         z_patches = crop_around_punkta(patch, pt[0], thresh_bck=thresh_bck, show=show_ignored, headless=headless)
#         if z_patches is not None:
#             [tifffile.imwrite(f"{patches_dir}/planes_{key}_z_{pt[0]}_y_{pt[1]}_x_{pt[2]}.tif",val.astype(np.float32)) for key,val in z_patches.items()]  
#             if show_accepted:
#                 print(f"Using z_{pt[0]}_y_{pt[1]}_x_{pt[2]}")
#                 show_patch(z_patches[list(z_patches.keys())[-2]])
#             accepted_counter = accepted_counter+1
#         else:
#             rejected_counter = rejected_counter + 1
#         processed_counter = processed_counter + 1

#     if not headless:
#         print(f"Processed {processed_counter} patches")
#         print(f"saved {accepted_counter} patches")
#         print(f"Rejected {rejected_counter} patches")

def find_psf_crops(img_path, crop_shape=(16,16,16), num_folders=50, thresh_bck=100, threshold_dist=10, unit = (1,1,1), show_ignored=False, show_accepted=False, headless=False):
    '''
    find, select and save patches
    img_path = String, path to the image
    crop_shape = 3d tuple of ints as (z,y,x). z axis is only for reference
    num_folders = int, for parallel processing, images will be saved in as many num_folders
    thresh_bck = float, absolute cut off value for SNR, depends on the images
    show_accepted, show_ignored = Boolean, flags for visualization of accepted or ignored patches
    headless = Boolean, displays output if False

    '''
    dirpath = os.path.dirname(img_path)
    img = tifffile.imread(img_path)

    if headless:
        show_ignored = False
        show_accepted = False

    # load points found in Fiji
    results_csv_path = dirpath+"/Results.csv"
    try:
        # fiji_points = genfromtxt(results_csv_path, delimiter=',')#.astype() #load the 2d points from scv file
        #
        # Jamie: not elegant but it gets the job done.
        #
        # Read data
        fiji_points = genfromtxt(results_csv_path, delimiter=',')#.astype(float) #load the 2d points from scv file
        # get ride of nans
        fiji_points = fiji_points[~np.isnan(fiji_points)]
        fiji_points = fiji_points.astype(int)
        fiji_points = fiji_points.reshape(int(len(fiji_points)/3),3)    
    except:
        print("Cannot load Results.csv file")
    if fiji_points.shape[-1]==3:
        points2d = fiji_points[1:,1:] # leave out the first row (contains strings X and Y ) and first column (contains enumeration of points)
    else:
        points2d = fiji_points[1:,:]  # leave out the first row (contains strings X and Y ) 
    points2d[:,[1,0]] = points2d[:,[0,1]] # flip x and y co-ordinates
    if not headless:
        print(f"Found {points2d.shape[0]} points")

    # throw out points that are very close determined by threhold_dist
    # threshold_dist = np.maximum(8,crop_shape[-1]/3)
    # threshold_dist = 10
    if threshold_dist is None:
        threshold_dist = 1.13/unit[-1]
    dist = (np.sort(cdist(points2d, points2d),1)[:,1:]<threshold_dist)[:,0]
    if not headless:
        print(f"Deleting {sum(dist==True)} because points are too close")
    points2d = np.delete(points2d, np.where(dist==True),0)

    # find the location of maximum pixel along z axis and create 3d points
    zmax = [np.argmax(img[:,_p[0],_p[1]]) for _p in points2d]
    points = np.insert(points2d, 0, zmax, 1)

    # crop patches around the 3d points
    new_points, patches = get_patches(points.copy(), img, patch_shape=tuple([int(t/2) for t in crop_shape]), headless=headless);

    processed_counter = 0
    accepted_counter = 0
    rejected_counter = 0

    # remove all the directories and make new ones
    patches_dirs = glob.glob(f"{dirpath}/patches_for_zola/patches_*/")
    [shutil.rmtree(p, ignore_errors=True) for p in patches_dirs]
    [os.makedirs(f"{dirpath}/patches_for_zola/patches_{i}/") for i in range(num_folders)]  
        
    # loop over the points and patches
    # for pt, patch in tqdm(zip(new_points, patches), disable=headless):
    for pt, patch in zip(new_points, patches):

        i = int(np.random.uniform(low = 0, high=num_folders))
        patches_dir = f"{dirpath}/patches_for_zola/patches_{i}/"

        # crop th patch around the punkta
        z_patches = crop_around_punkta(patch, pt[0], thresh_bck=thresh_bck, show=show_ignored, headless=headless)
        if z_patches is not None:
            [tifffile.imwrite(f"{patches_dir}/planes_{key}_z_{pt[0]}_y_{pt[1]}_x_{pt[2]}.tif",val.astype(np.float32)) for key,val in z_patches.items()]  
            if show_accepted:
                print(f"Using z_{pt[0]}_y_{pt[1]}_x_{pt[2]}")
                show_patch(z_patches[list(z_patches.keys())[-2]])
            accepted_counter = accepted_counter+1
        else:
            rejected_counter = rejected_counter + 1
        processed_counter = processed_counter + 1

    if not headless:
        print(f"Processed {processed_counter} patches")
        print(f"saved {accepted_counter} patches")
        print(f"Rejected {rejected_counter} patches")

def load_zola_files(zola_raw_dir, lam=0.515, max_amp=None, headless=False):
    '''
    load all the zola files in the given directory
    zola_raw_dir = String, directory containing zola_raw folder
    lam = float, wavelength
    max_amp = float, maximum amplitude allowed, default None then all values are allowed
    headless = Boolean, displays output if False

    returns a dictionary of zola files each key having ansi ordered aberration amplitude values
    '''

    zola_files = glob.glob(f"{zola_raw_dir}/zola_raw/*.json")
    if not headless:
        print(f"Found {len(zola_files)} zola files")
    names = [os.path.splitext(os.path.basename(p))[0] for p in zola_files]
    
    zola_list_full = OrderedDict()

    for n,z in zip(names, zola_files):
        zola_res,flag = zola_processed(z,lam=lam, max_amp=max_amp)
        del zola_res['0'], zola_res['1'], zola_res['2'], zola_res['4'] # delete piston, tip, tilt, defocus
        if flag==0:
            continue
        zola_list_full[n] = zola_res;
    return zola_list_full

def load_and_select_zola_files(img_dir, lam, max_dist = 0.06, headless=False, show_plots=False):
    '''
    load all the zola files in the given directory and select the ones that are stable
    img_dir = String, directory containing patches_for_zola folder
    lam = float, wavelength
    headless = Boolean, displays output if False

    returns a dictionary of zola files each key having ansi ordered aberration amplitude values
    '''

    # load all the files
    multi_plane_zola = load_zola_files(img_dir+"/patches_for_zola/", lam=lam, headless=headless)
    names = np.unique([n[n.find('z_'):] for n in multi_plane_zola.keys()])
    if not headless:
        print(f"Processing {len(multi_plane_zola)} files")

    zola_full = OrderedDict()
    planes = np.arange(14,9,-2) # planes used
    del_keys = []

    for key in tqdm(names):
        val = []

        # load the zola results for all the planes
        for i,pl in enumerate(planes):
            val.append(np.array(list(multi_plane_zola[f"planes_{pl}_{key}"].values())))
        val = np.array(val)
        
        # select the zola files whoose results are stable
        dist = np.max(cdist(val,val))
        if dist>max_dist:
            [del_keys.append(f"planes_{pl}_{key}") for i,pl in enumerate(range(10,1,-2))]
            if not headless:
                print(f"NOT Using {key}")
        else:
            zola_full[key] = {k:v for k,v in zip(multi_plane_zola[f"planes_{pl}_{key}"].keys(),np.mean(val,0))}
            if not headless:
                print(f"Using {key} with rmswe {np.sqrt(np.sum(np.array(list(zola_full[key].values()))**2))}")
        
        if not headless:
            print(f"{key} has a maximum distance of {dist} between all its zola files")

        if show_plots:
            import seaborn as sns
            import pandas as pd
            fig, ax = plt.subplots(1,1,figsize=(4,4));           
            p1 = pd.DataFrame(data=val, columns=range(val.shape[-1]))
            df_c = pd.melt(p1)
            df_c['planes'] = planes.tolist()* val.shape[-1]
            sns.barplot(data =df_c, x='variable',y='value',hue='planes', dodge=True)
            plt.legend(title="Planes", frameon=False)
            plt.xticks(range(len(val[i])))
            plt.ylim(-0.14,0.14)
            plt.tight_layout()
            plt.show()

    if not headless:
        print(f"Deleting {len(del_keys)} files")

    return zola_full 

def twoD_GaussianScaledAmp(xy, xo, yo, sigma_x, sigma_y, amplitude, offset, theta):
    """
    Function to fit, returns 2D gaussian function as 1D array
    https://gist.github.com/nvladimus/fc88abcece9c3e0dc9212c2adc93bfe7
    """
    (x, y) = xy
    xo = float(xo)
    yo = float(yo)    

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def getFWHM_GaussianFitScaledAmp(img, show_fit=False, ax=None):
    """
    Get FWHM(x,y) of a blob by 2D gaussian fitting
    https://gist.github.com/nvladimus/fc88abcece9c3e0dc9212c2adc93bfe7
    Parameter:
        img - 2d image as numpy array
    Returns: 
        FWHMs in pixels, along x and y axes.
    """

    x = np.linspace(0, img.shape[1]-1, img.shape[1])
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline, theta
    maxi_pt = np.unravel_index(np.argmax(img),img.shape)
    initial_guess = (maxi_pt[1], maxi_pt[0],10,10,1,0,0)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img,5)
    img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
    popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, (x, y), 
                               img_scaled.ravel(), p0=initial_guess)
    
    if not np.sqrt(np.diag(pcov))[0]<1:
        raise ValueError

    data_fitted = twoD_GaussianScaledAmp((x, y), *popt)
    
    if show_fit:
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.imshow(img)
        ax.contour(x, y, data_fitted.reshape(img.shape[0], img.shape[1]), 8, colors='w')
        # plt.show()

    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    return (FWHM_x, FWHM_y)


def select_files_by_med_dist_of_neighbors(zola_full, mode_num_for_median_disp=12, diff_thresh = 0.06, scale_factor=4):
    """
    Use clesperanto to filter out zola results which deviate from median of neighbors
    Parameter:
    zola_full : dictionary of zola files, the key of the dictionary should be z_XX_y_XX_x_XX
    mode_num_for_median_disp : 'ansi' ordered zernike index
    diff_thresh : difference allowed
    scale_factor : scaling of the points
    
    Returns: list of keys to be deleted 
        
    """
    try:
        import pyclesperanto_prototype as cle
    except:
        print("cant load cles")    


    # scaled known points
    point_list = np.array([np.divide(list(map(int, re.findall(r'\d+', key))), (scale_factor,)) for key in zola_full.keys()])
    ori_point_list = np.array([list(map(int, re.findall(r'\d+', key))) for key in zola_full.keys()])
    point_list[:,[0,1,2]] = point_list[:,[2,1,0]] # changing x and z for pyclesperanto
    point_list = point_list.T

    # label the pixels whose value is known
    # fix point_list array September 2023 jw
    # labeled_pixel = cle.pointlist_to_labelled_spots(cle.push(point_list.T))
    labeled_pixel = cle.pointlist_to_labelled_spots(cle.push(point_list))

    # extend the label by vornoi
    labeled_image = cle.extend_labeling_via_voronoi(labeled_pixel)

    # create a touch matrix
    tm = cle.generate_touch_matrix(labeled_image)

    # measurements
    zola_list = np.array([_zo[str(mode_num_for_median_disp)] for _zo in zola_full.values()])
    zola_list = np.concatenate([[0],zola_list]) # adding the measurement at the beginning for label index
    measurement = cle.push(np.asarray(np.expand_dims(np.expand_dims(zola_list,0),0)).T)
    med_measurement = cle.median_of_touching_neighbors(measurement, tm)
    diff = cle.absolute_difference(measurement, med_measurement)[0,0]
    med_del_index = np.argwhere(diff>diff_thresh).flatten()-1
    med_del_keys = [f'z_{ori_point_list[i][0]}_y_{ori_point_list[i][1]}_x_{ori_point_list[i][2]}' for i in med_del_index]
    return med_del_keys

def filter_files_by_med_dist_of_neighbors(zola_full, abb_list=[3,5,6,7,8,9,10,11,12,13,14], diff_thresh=0.1, headless=False):
    '''
    wrapper around the function select_files_by_med_dist_of_neighbors for all the zern amplitudes
    zola_full : dictionary of zola files, the key of the dictionary shoudl be z_XX_y_XX_x_XX
    diff_thresh : difference allowed
    '''
    median_del_keys = []

    for i in abb_list:
        med_del_key = select_files_by_med_dist_of_neighbors(zola_full, mode_num_for_median_disp=i,diff_thresh=diff_thresh)
        median_del_keys = median_del_keys+med_del_key
    return np.unique(np.array(median_del_keys))

def get_xyz_widths(crops, unit, headless=True):
    '''
    get FWHM in xy and yz
    crops : dictionary of patches, the key of the dictionary should be z_XX_y_XX_x_XX

    returns a dictionary of fwhm
    '''
    fwhm = OrderedDict()
    for key,val in tqdm(crops.items(), disable=headless):
        _, yy, xx = val.shape
        zz = np.argmax(val[:,yy//2,xx//2])
        if not headless:
            print(key)
            fig, axes = plt.subplots(ncols=2, nrows=1)
        try:
            if not headless:
                fwhm_x, fwhm_y = getFWHM_GaussianFitScaledAmp(val[zz], show_fit=True, ax=axes[0])
            else:
                fwhm_x, fwhm_y = getFWHM_GaussianFitScaledAmp(val[zz], show_fit=False, ax=None)
        except:
            if not headless:
                print("Can't fit the xy axis")
            continue
        try:
            if not headless:
                _, fwhm_z = getFWHM_GaussianFitScaledAmp(val[:,yy//2], show_fit=True, ax=axes[1])
            else:
                _, fwhm_z = getFWHM_GaussianFitScaledAmp(val[:,yy//2], show_fit=False, ax=None)
        except:
            if not headless:
                print("Can't fit the zy axis")
            continue
        fwhm[key] = np.abs(np.array([fwhm_z, fwhm_y, fwhm_x])*np.array(unit))
        if not headless:
            plt.show()

    return fwhm

# def create_abb_map(img_path, zola_full, abb_list=[3,5,6,7,8,9,10,11,12,13,14], save_dir=None, scale_factor=(4,)*3, dont_scale_maksed_img = False):
def create_abb_map(img_path, zola_full, abb_list=[3,5,6,7,8,9,10,11,12,13,14], save_dir=None, scale_factor=(4,4,4), dont_scale_maksed_img = False):

    try:
        import pyclesperanto_prototype as cle
    except:
        print("cant load cles")    

    if save_dir is None:
        save_dir = os.path.dirname(img_path)

    # Delete previous maps
    ansi_files = glob.glob(f"{save_dir}/ansi_*.tif")
    [os.remove(f) for f in ansi_files];

    # load a masked image if availbale
    try:
        masked_img = tifffile.imread(os.path.join(os.path.dirname(img_path), f"*masked*.tif"))
        if not dont_scale_maksed_img:
            masked_img = downscale_local_mean(masked_img, scale_factor)
    except:
        print("No masked image found")
        img = tifffile.imread(img_path)
        masked_img = downscale_local_mean(img, scale_factor)

    # show and save the acce[ted points
    fig = plt.figure()  
    plt.imshow(np.max(masked_img,0), cmap='rocket', clim=(np.percentile(masked_img,3), np.percentile(masked_img, 97)))
    pos = np.array([np.divide(list(map(int, re.findall(r'\d+', n))), scale_factor) for n in zola_full.keys()])
    plt.plot(pos[:,2], pos[:,1],'g.')
    plt.suptitle(f"Accepted Points")
    plt.show()
    plt.savefig(f"{save_dir}/accepted_points_in_image.png")

    # scale known points
    point_list = np.array([np.divide(list(map(int, re.findall(r'\d+', key))), scale_factor) for key in zola_full.keys()])
    point_list = np.concatenate((point_list, ([[_s-1 for _s in masked_img.shape]])), axis=0) # adding a point right at the end
    point_list[:,[0,1,2]] = point_list[:,[2,1,0]] # changing x and z for pyclesperanto
    point_list = point_list.T

    # label the pixels whose value is known
    # need to fix arrays (Sept 2023)
    # labeled_pixel = cle.pointlist_to_labelled_spots(cle.push(point_list.T))
    labeled_pixel = cle.pointlist_to_labelled_spots(cle.push(point_list))


    # extend the label by vornoi
    labeled_image = cle.extend_labeling_via_voronoi(labeled_pixel)
    maxi_image = cle.maximum_box(labeled_pixel,radius_x=4, radius_y=4, radius_z=0)

    # create a touch matrix
    tm = cle.generate_touch_matrix(labeled_image)

    # save the map
    tifffile.imwrite(f"{save_dir}/label_map.tif", np.array(labeled_image))
    np.save(f"{save_dir}/pointlist.npy", np.array(point_list))

    dirs = ['abb_maps','point_boxes','points','measurements','med_measurement','rmswe']
    [shutil.rmtree(f"{save_dir}/{i}/", ignore_errors=True) for i in dirs];
    [os.makedirs(f"{save_dir}/{i}/") for i in dirs];

    for a in abb_list:
        # measurements
        zola_list = np.array([_zo[str(a)] for _zo in zola_full.values()])
        zola_list = np.concatenate([[0],zola_list,[0]]) # adding the measurement of the end point and beginning for label index
        measurement = cle.push(np.asarray(np.expand_dims(np.expand_dims(zola_list,0),0)).T)
        
        
        # # replace the vornoi with the measurements
        # parametric_image = cle.replace_intensities(labeled_image, measurement)
        # maxi_parametric_image = cle.replace_intensities(maxi_image, measurement)
        # maxi_parametric_image_points = cle.replace_intensities(labeled_pixel, measurement)
        parametric_image = cle.replace_intensities(labeled_image, measurement.ravel())
        maxi_parametric_image = cle.replace_intensities(maxi_image, measurement.ravel())
        maxi_parametric_image_points = cle.replace_intensities(labeled_pixel, measurement.ravel())

        # replace the vornoi with the median measurements of touching neighbors to remove noise
        med_measurement = cle.mean_of_touching_neighbors(measurement, tm)
        # med_parametric_image = cle.replace_intensities(labeled_image, med_measurement)
        med_parametric_image = cle.replace_intensities(labeled_image, med_measurement.ravel())
        
        # rescale and masking
        show_image = np.array(parametric_image)
        show_image[masked_img==0] = np.nan
        show_image_1 = np.array(med_parametric_image)
        show_image_1[masked_img==0] = np.nan
        
        
        #show
        contrast_lim = 0.1
        fig, axes = plt.subplots(1,2, figsize=(10,4))
        im = axes[0].imshow(np.nanmean(show_image,0), cmap='seismic',clim=(-1*contrast_lim,contrast_lim))
        axes[1].imshow(np.nanmean(show_image_1,0), cmap='seismic', clim=(-1*contrast_lim,contrast_lim))
        [axes[i].axis('OFF') for i in range(2)]
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
        plt.suptitle(f"ANSI {a}")
        plt.show()

        #save
        tifffile.imwrite(f"{save_dir}/abb_maps/ansi_{a}.tif", np.array(show_image)) 
        tifffile.imwrite(f"{save_dir }//abb_maps/smoothened_ansi_{a}.tif", np.array(show_image_1)) 
        tifffile.imwrite(f"{save_dir }/point_boxes/ansi_{a}.tif", np.array(maxi_parametric_image)) 
        tifffile.imwrite(f"{save_dir }/points/ansi_{a}.tif", np.array(maxi_parametric_image_points)) 
        np.save(f"{save_dir}/measurements/ansi_{a}.npy", np.array(measurement))
        np.save(f"{save_dir}/med_measurement/ansi_{a}.npy", np.array(med_measurement))

def find_rmswe(abb_maps_dir, abb_list=[3,5,6,7,8,9,10,11,12,13,14]):
    abb_maps = np.array([tifffile.imread(f"{abb_maps_dir}/abb_maps/smoothened_ansi_{a}.tif") for a in abb_list])
    modes_indexes = [[3,5,6,7,8,9], [10,11,12,13,14], abb_list]

    for modes_index in modes_indexes:
        modes = [abb_list.index(ele) for ele in modes_index]
        rmswe = np.sqrt(np.nansum(abb_maps[modes]**2, axis=0))
        plt.title(f"{modes_index}")
        plt.imshow(np.nanmax(rmswe,0), cmap='rocket_r',clim=(0.0,0.2))
        plt.colorbar();
        plt.axis('OFF');
        tifffile.imwrite(f"{abb_maps_dir}/rmswe/{modes_index}.tif", np.array(rmswe))
        plt.show();


