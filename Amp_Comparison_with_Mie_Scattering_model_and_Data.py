import sys
from astropy.io import ascii
import numpy as np
import matplotlib as mpl
from matplotlib import rc
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.convolution import convolve
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.stats import chisquare
import pdb
import os
import astropy.io.fits as fits
import matplotlib.lines as lines
# from blessings import Terminal

def fourier_sinusoidal_series_oneMode(x, a0,b0,a1,b1,freq, offset):
    # sin(alpha + beta) = sin(alpha) * cos(beta) + cos(alpha) * sin(beta)... so on depending how many in a series
    n_two = 2.
    result = a0*np.cos(2.*0*np.math.pi*freq*x) \
             + b0*np.sin(2.*0*np.math.pi*freq*x) \
             + a1*np.cos(2.*1*np.math.pi*freq*x) \
             + b1*np.sin(2.*1*np.math.pi*freq*x) \
             + offset
    return result
def fourier_sinusoidal_series_twoModes(x, a0,b0,a1,b1,a2,b2, freq, offset):
    result = a0*np.cos(2.*0*np.pi*freq*x) \
             + b0*np.sin(2.*0*np.pi*freq*x) \
             + a1*np.cos(2.*1*np.pi*freq*x) \
             + b1*np.sin(2.*1*np.pi*freq*x) \
             + a2*np.cos(2.*2*np.pi*freq*x) \
             + b2*np.sin(2.*2*np.pi*freq*x) \
             + offset
    return result
# fourier_sinusoidal_series function with modes ranging from 3 to ll modes
def time_converter(time_data, sptzr=True):
    #converting bmjd to hours
    if sptzr:
        day_in_hrs = 24
        extra_time = 5.71793e4
        hour = (time_data -extra_time) * day_in_hrs
        # times = np.array(hour)
    else:
        day_in_hrs = 24
        extra_time = 5.71793e4
        hour = (time_data - extra_time) * day_in_hrs
        # times = np.array(hour)
    return hour
def clip_of_mask_flux(clip_flux):
    clip = sigma_clip( clip_flux, sigma=5, sigma_lower=None, sigma_upper=None, iters=5, cenfunc=np.ma.median, stdfunc=np.std, axis=None, copy=True)
    clipped = clip[np.logical_not(clip.mask)] - 1.
    return clipped
def clip_of_mask (clp_med_flux, x):
    clip = sigma_clip(clp_med_flux, sigma=5, sigma_lower=None, sigma_upper=None, iters=5, cenfunc=np.ma.median, stdfunc=np.std, axis=None, copy=True)
    clipped_x = x[np.logical_not(clip.mask)]
    return clipped_x
def med_flux (x):
    # Reducing the data flux to it's median
    median = np.median(x)  # nan meaning that it does the calc the median but ignores nans
    x_divide_med = x / median
    med_x = np.array(x_divide_med)
    return med_x
# convolve function (not in use)
def convolve_data(flux):
    # smoothing the plot
    # best used if you already clipped the flux
    nbin = 5
    conv_flux = np.convolve(flux,(np.ones(nbin) / nbin),mode='same')
    return conv_flux
def amp_funct(x):
    min_value = min(x)
    max_value = max(x)
    med_value = np.median(x)
    ans = ((max_value - min_value) / med_value)
    # diff = (max_value - min_value)
    return ans
def relativeChange_for_oneModel (model_one):
    min_v = min(model_one)
    max_v = max(model_one)
    abs_change_min =max_v - min_v
    relative_change = (abs_change_min / min_v)*100
    return relative_change
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------- List of directories for HST and SPTZ ---------------------------------------

sptz_data_directory = '/Users/melaniapena/Rsrch/Data/SpitzerData/SPITZER_PHOTOMETRY_1629+03.txt'
hst_data_directory = '/Users/melaniapena/Rsrch/Data/HST_data/virtualdatanew.txt'

# Once the fourier fit and data is plotted onto the same graph, the figure will be saved into a directory.
save_figure = '/Users/melaniapena/Rsrch/code/2017_fall_semester/data/Int_Amp_Dec_21_17/HST_SPTZ_fit_and_ori_data.png'
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------- Read in Ascii ----------------------------------------------------

# SPITZER DATA written as an ascii file
data_SPTZ = ascii.read(sptz_data_directory)

# depending what was written in the ascii file
bmjd_SPTZ = data_SPTZ['BMJD']
flux_SPTZ = data_SPTZ['FLUX']
flux_err_SPTZ = data_SPTZ['FLUX_ERR']
x_cen_SPTZ = data_SPTZ['X_CENTROID']
y_cen_SPTZ = data_SPTZ['Y_CENTROID']

# HUBBLE SPACE DATA written as an ascii file
data_HST = ascii.read(hst_data_directory)

# depending what was written in the ascii file
bmjd_HST = data_HST['Midtime']
J_band_flux_HST = data_HST['J_band(ergs/(s*cm^2))']
J_band_flux_err_HST = data_HST['JBandError']
H_band_flux_HST = data_HST['H_band(ergs/(s*cm^2))']
H_band_flux_err_HST = data_HST['HBandError']
W_band_flux_HST = data_HST['W_band(ergs/(s*cm^2))']
W_band_flux_err_HST = data_HST['WBandError']

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- SPTZR Data -----------------------------------------------------
# reduce data flux and time converter to hours
# flux_med_SPTZ = med_flux(flux_SPTZ)
hour_SPTZ = time_converter(bmjd_SPTZ)
flux_med_SPTZ = (flux_SPTZ / np.nanmedian(flux_SPTZ))
flux_err_med_SPTZ =  (flux_err_SPTZ / np.nanmedian(flux_SPTZ))
# clip the original data using the flux data points
clip_flux_SPTZ =clip_of_mask_flux(flux_med_SPTZ)
clip_flux_err_SPTZ =clip_of_mask(flux_med_SPTZ,flux_err_med_SPTZ)
clip_hour_SPTZ = clip_of_mask(flux_med_SPTZ,hour_SPTZ)
# convolve the clipped data
conv_clip_flux_SPTZ = convolve_data(clip_flux_SPTZ)

# stretches the graph but does not affect the x and y axis
plt.figure(figsize=(10.5,7))

# ---------------------- Plotting the Fit and original clipped Data ----------------------
# original plot SPITZER data
plt.plot(clip_hour_SPTZ,clip_flux_SPTZ+1,'o',color='xkcd:cerulean blue',markersize=0.8,label='Spitzer Data')

# smooths out the data of Spitzer
# plt.plot(clip_hour_SPTZ,conv_clip_flux_SPTZ+1,'o',color='xkcd:cerulean blue',markersize=0.8,label='Spitzer Data')

# ----------------------  SPTZ data using two mode fourier series ----------------------
# fit of the original plotted data using fourier series
guess_twoMode=[1./5.,1./5.,1./5.,1./5.,1./5.,1./5.,0.1,0.]
params_SPTZ_twoMode, pcov = curve_fit(fourier_sinusoidal_series_twoModes, clip_hour_SPTZ, clip_flux_SPTZ, p0=guess_twoMode, sigma=clip_flux_err_SPTZ)
SPTZ_plot = plt.plot(clip_hour_SPTZ,fourier_sinusoidal_series_twoModes(clip_hour_SPTZ,*params_SPTZ_twoMode)+1,'-',  color='xkcd:cerulean blue', linewidth = 1,label = 'Spitzer Fit')

# plt.legend(loc=2)

#  ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ HST Data ------------------------------------------------------
# Converting julian into hours and since hst data has gap data wee fill in the gaps with numpy linespace
times_hst = time_converter(bmjd_HST)
times_int_hst = np.linspace(min(times_hst), max(times_hst),num=80)

# Setting the HST data to Zero so it aligns with the SPITZER data
J_flux_med_HST = (J_band_flux_HST/ np.median(J_band_flux_HST))
H_flux_med_HST = (H_band_flux_HST / np.median(H_band_flux_HST))
W_flux_med_HST = (W_band_flux_HST / np.median(W_band_flux_HST))

J_flux_err_med_HST =  (J_band_flux_err_HST/ np.median(J_band_flux_HST))
H_flux_err_med_HST =  (H_band_flux_err_HST/ np.median(H_band_flux_HST))
W_flux_err_med_HST =  (W_band_flux_err_HST/ np.median(W_band_flux_HST))


#----Plotting the Fit and original clipped Data ----
# Original data plot
plt.plot(times_hst,J_flux_med_HST,'or',markersize=1.5,label='Hubble J-Band Data')
plt.plot(times_hst,H_flux_med_HST,'o',color='xkcd:dark seafoam green',markersize=1.5,label='Hubble H-Band Data')
plt.plot(times_hst,W_flux_med_HST,'o',color='xkcd:sunflower',markersize=1.5,label='Hubble W-Band Data')
# --------------------------------------------------

# note: keep original times data in the params variable. it is fitting the original. not the np.linspace
# ------- HST data using two mode fourier series -------
guess_twoMode=[1./5.,1./5.,1./5.,1./5.,1./5.,1./5.,0.1,0.]
# Finds the fourier fit of the HST and Spitzer Data
params_J_HST_twoMode, pcov = curve_fit(fourier_sinusoidal_series_twoModes, times_hst, J_flux_med_HST, p0=guess_twoMode, sigma=J_flux_err_med_HST)
params_H_HST_twoMode, pcov = curve_fit(fourier_sinusoidal_series_twoModes, times_hst, H_flux_med_HST, p0=guess_twoMode, sigma=H_flux_err_med_HST)
params_W_HST_twoMode, pcov = curve_fit(fourier_sinusoidal_series_twoModes, times_hst, W_flux_med_HST, p0=guess_twoMode, sigma=W_flux_err_med_HST)
# plots the fourier fit models
HST_J_plot, = plt.plot(times_int_hst,fourier_sinusoidal_series_twoModes(times_int_hst,*params_J_HST_twoMode),'r', linewidth = 1,label = 'Hubble J-Band Fit')
HST_H_plot, = plt.plot(times_int_hst,fourier_sinusoidal_series_twoModes(times_int_hst,*params_H_HST_twoMode),'xkcd:dark seafoam green', linewidth = 1,label = 'Hubble H-Band Fit')
HST_W_plot, = plt.plot(times_int_hst,fourier_sinusoidal_series_twoModes(times_int_hst,*params_W_HST_twoMode),'xkcd:sunflower', linewidth = 1,label = 'Hubble W-Band Fit')

text_title_2 = "Observation of HST and Spitzer"
plt.title(text_title_2 ,fontsize = 14)
plt.xlabel('Time (hrs)', fontsize = 14)      # labeling Figure
plt.ylabel('Photometric Flux',fontsize = 14)


# This creates the list within the legend of the graph.
J_band_line = lines.Line2D([], [], color='red',linestyle='None', marker='.', markersize=5, label='HST/WFC3 J Band')
H_Band_line = lines.Line2D([], [], color='xkcd:dark seafoam green', linestyle='None', marker='.', markersize=5, label='HST/WFC3 H Band')
W_Band_line = lines.Line2D([], [], color='xkcd:sunflower',linestyle='None', marker='.', markersize=5, label='HST/WFC3 Water Band')
SPTZ_line = lines.Line2D([], [], color='xkcd:cerulean blue',linestyle='None', marker='.', markersize=5, label='Spitzer [3.6]')
J_band_line_fit = lines.Line2D([], [], color='red', marker='', markersize=5, label='J Band Fit')
H_Band_line_fit = lines.Line2D([], [], color='xkcd:dark seafoam green', marker='', markersize=5, label='H Band Fit')
W_Band_line_fit = lines.Line2D([], [], color='xkcd:sunflower', marker='', markersize=5, label='Water Band Fit')
SPTZ_line_fit = lines.Line2D([], [], color='xkcd:cerulean blue', marker='', markersize=5, label='[3.6] Fit')

# The legend itself
# lgd=plt.legend(handles=[J_band_line,H_Band_line,W_Band_line,SPTZ_line, J_band_line_fit,H_Band_line_fit,W_Band_line_fit,SPTZ_line_fit],loc=2, ncol=2)
lgd=plt.legend(handles=[J_band_line,H_Band_line,W_Band_line,SPTZ_line, J_band_line_fit,H_Band_line_fit,W_Band_line_fit,SPTZ_line_fit],loc=2,bbox_to_anchor=(0.1, -0.13), ncol=2)
art = []
art.append(lgd)
# Sets the size range of the graph
plt.xlim(0,9)
plt.ylim(0.97,1.06)

# Saves the figure into a directory.
plt.savefig(save_figure,dpi=600, additional_artists=art, bbox_inches="tight")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------


#   The coding in the "Overlap plot region" just shows the 'only' the overlapped portion of the same graph that
# presented as the HST and Spitzer data. the overlap plot will be commented out unless needed.


# ------------------------------------ Overlap plot region------------------------------------------
# Measure the the amplitudes for each band separately by taking the max and min of the model.
# but when comparing with the Spitzer data, it doesnt quite overlap the whole data model of the HST band models.
# what we want to be able to compare the amplitudes only within regions that overlap.

# find the end and points between the overlap data with both SPITZR and HST bands
startPoint_SPTZ = clip_hour_SPTZ[0]
lastPoint_HST_J = times_int_hst[-1]
lastPoint_HST_H = times_int_hst[-1]
lastPoint_HST_W = times_int_hst[-1]

# Using the where function and if else statement to only have an overlap region of the graph
hst_J_index = np.where(np.logical_and(times_int_hst >= startPoint_SPTZ, times_int_hst <= lastPoint_HST_J))
hst_H_index = np.where(np.logical_and(times_int_hst >= startPoint_SPTZ, times_int_hst <= lastPoint_HST_H))
hst_W_index = np.where(np.logical_and(times_int_hst >= startPoint_SPTZ, times_int_hst <= lastPoint_HST_W))
sptz_index = np.where(np.logical_and(clip_hour_SPTZ >= startPoint_SPTZ, clip_hour_SPTZ <= lastPoint_HST_J))


# plots the overlap data and fourier fit of the HST and Spitzer
'''
plt.title('Overlap Region of HST/SPTZ Amp Data')
plt.xlabel('Wavelength')      # labeling Figure
plt.ylabel('Photometric Flux')

# HST, Spitzer data.
plt.plot(clip_hour_SPTZ[sptz_index],(clip_flux_SPTZ+1)[sptz_index], '.', linewidth = .01,label = 'SPTZ')
plt.plot(times_int_hst[hst_J_index], J_flux_med_HST[hst_J_index], '.', linewidth = 1, label ='J Band')
plt.plot(times_int_hst[hst_H_index], H_flux_med_HST[hst_H_index], '.', linewidth = 1, label ='H Band')
plt.plot(times_int_hst[hst_W_index], W_flux_med_HST[hst_W_index],  '.', linewidth = 1, label ='W Band')

# Overlapped Fourier Fit 
plt.plot(clip_hour_SPTZ[sptz_index],(fourier_sinusoidal_series_twoModes(clip_hour_SPTZ,*params_SPTZ_twoMode)+1)[sptz_index],"b", linewidth = 1,label = 'SPTZ')
plt.plot(times_int_hst[hst_J_index], fourier_sinusoidal_series_twoModes(times_int_hst, *params_J_HST_twoMode)[hst_J_index], "r", linewidth = 1, label ='J Band')
plt.plot(times_int_hst[hst_H_index], fourier_sinusoidal_series_twoModes(times_int_hst, *params_H_HST_twoMode)[hst_H_index], "g", linewidth = 1, label ='H Band')
plt.plot(times_int_hst[hst_W_index], fourier_sinusoidal_series_twoModes(times_int_hst, *params_W_HST_twoMode)[hst_W_index], "y", linewidth = 1, label ='W Band')
plt.show()
'''

# ---------------------------------------Finding Amplitude of Overlapped Region-----------------------------------------
sptz_overlap = ((fourier_sinusoidal_series_twoModes(clip_hour_SPTZ,*params_SPTZ_twoMode)+1)[sptz_index])
hst_J_overlap =fourier_sinusoidal_series_twoModes(times_int_hst, *params_J_HST_twoMode)[hst_J_index]
hst_H_overlap =fourier_sinusoidal_series_twoModes(times_int_hst, *params_H_HST_twoMode)[hst_H_index]
hst_W_overlap = fourier_sinusoidal_series_twoModes(times_int_hst, *params_W_HST_twoMode)[hst_W_index]

# Calculate its Amplitude Region
amp_J_HST = amp_funct(hst_J_overlap)
amp_H_HST = amp_funct(hst_H_overlap)
amp_W_HST = amp_funct(hst_W_overlap)
amp_SPTZ = amp_funct(sptz_overlap)

# Sets the Wavelengths that goes along with the calculated amplitudes
J_wavelength = 1.27
H_wavelength = 1.60
W_wavelength = 1.4
Sptz_wavelength = 3.6

# Creates the values as an array. When plotted, it will be Wavelength vs. Amplitude
Wavelengths_of_HST_SPTZ = [J_wavelength, W_wavelength, H_wavelength, Sptz_wavelength]
Amp_of_HST_SPTZ = [amp_J_HST,amp_W_HST,amp_H_HST,amp_SPTZ]
# print(Amp_of_HST_SPTZ)

# Prints the values of the Amplitudes of the HST and SPTZER data
print "HST J Amp: ",amp_J_HST, "\nHST H Amp: ",amp_H_HST, "\nHST W Amp: ",amp_W_HST, "\nSPTZ Amp: ",amp_SPTZ,"\n\n"

# Creates an array just for the HST data (used for the for-loop when comparing the amplitude with a Mie Scattering Model.)
Hst_bands =([amp_J_HST, amp_H_HST, amp_W_HST])


# plots wavelength vs. amplitude
'''
plt.title('Overlap Region of HST/SPTZ Amp Data')
plt.xlabel('Wavelength')      # labeling Figure
plt.ylabel('Amplitude')

# Individual scatter plot
plt.plot(J_wavelength,amp_J_HST,'or')
plt.plot(H_wavelength,amp_H_HST,'og')
plt.plot(W_wavelength,amp_W_HST,'oy')
plt.plot(Sptz_wavelength,amp_SPTZ,'ob')

# plotted as a line plot.
plt.plot(Wavelengths_of_HST_SPTZ, Amp_of_HST_SPTZ)

## The overlapped data points for both the HST and Spitzer
# print ((fourier_sinusoidal_series_twoModes(clip_hour_SPTZ,*params_SPTZ_twoMode)+1)[sptz_index])
# print (fourier_sinusoidal_series_twoModes(times_int_hst,*params_J_HST_twoMode)[hst_J_index])
# pdb.set_trace()

plt.show()
'''
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------- Computing the BIC Value and Chi-Squared --------------------------------
# calculates chi squared
# Mi is the model used from the data (the model fit of the original data)
# Yi is the original array of data points used (original data)
# sigma_error_i is the error points of the original data (original error data)
def chi_squared_GOF(Mi, Yi, sigma_error_i):
    # the sum of the value minus of the expected value divided by the sigma of the expected all of it is squared
    # Chi-Squared (Goodness of Fit)
    # chi_squared = sum((Yi - Mi) ** two) / sum(sigma_error_i**two)
    two = 2
    chi_squared = sum(((Yi - Mi) ** two) / sigma_error_i ** two)
    return chi_squared
# calculates the Bayesion Information Criterion (BIC) and
# takes in the parameters such as...
# - k (the number of params in a model),
# - Num_points (the amount of points that is in within an array of data),
# - chi_sqrd (the result you received from the chi_sqrd_GOF function)
def BIC(k, Num_points, chi_sqrd):
    # Bayesian Information Criterion
    # K is for the parameters (params )
    bic = (k * np.log(Num_points)) + (chi_sqrd)
    return bic
# calls the bic and chi_squared_GOF functions in and prints out the results
# this def is used to label each result and keep track which mode (how many modes is used from the Fourier Series) is used
def num_of_modes(band_str,mode_str,fucntion,time,params,flux,fluxError,length,k):
    chi_sqrd = chi_squared_GOF(fucntion(time, * params), flux, fluxError)
    bic = BIC(k, length, chi_sqrd)
    print "Chi-Squared of {} flux in HST ({} Mode): ".format(band_str,mode_str),chi_sqrd
    print "BIC {} ({} Mode):                        ".format(band_str,mode_str),bic, "\n"
    return chi_sqrd,bic

# higher mode fourier series (2 modes - 11 modes), Originally written to check if we can use just a one term or a higher
# term and see if our final results changes as we added a new term. But in the end,
# we just used the second fourier series term.

# two mode fourier series
# using the amount of parameters (does not include first parameter) from the fourier series definition
k_twoMode = 8.

# prints out both the BIC and Chi-Squared
'''
# mode_two_J = num_of_modes("J","two", fourier_sinusoidal_series_twoModes, times_hst, params_J_HST_twoMode, J_flux_med_HST, J_flux_err_med_HST, len_of_HST_flux, k_twoMode)
# mode_two_H = num_of_modes("H","two", fourier_sinusoidal_series_twoModes, times_hst, params_H_HST_twoMode, H_flux_med_HST, H_flux_err_med_HST, len_of_HST_flux, k_twoMode)
# mode_two_W = num_of_modes("W","two", fourier_sinusoidal_series_twoModes, times_hst, params_W_HST_twoMode, W_flux_med_HST, W_flux_err_med_HST, len_of_HST_flux, k_twoMode)
# mode_two_SPTZ = num_of_modes("SPTZ","two",fourier_sinusoidal_series_twoModes,clip_hour_SPTZ,params_SPTZ_twoMode,clip_flux_SPTZ,clip_flux_err_SPTZ,len_of_SPTZ_flux,k_twoMode)
'''

# ----------------------------------------------------------------------------------------------------------------------

# the amount of data points is in an array in each HST band and SPTZ
len_of_HST_flux = len(J_flux_med_HST)  # 80 of the original data points
                                       #
                                       # note: all the flux bands (J,H, and W) in the HST
                                       #    have all the same amount of data points in each
                                       #    array. Which is why I only placed the J_flux band
                                       #    as the length of the HST only.
                                       #    it as to keep track which length was from SPTZ or HST..

len_of_SPTZ_flux = len(clip_flux_SPTZ) # 1005 of the clipped original data
                                       #
                                       # note: the original data of SPTZ contained 1037 but was
                                       #    not included because it was considered noise due to the data
                                       #    points being too far from the majority and was considered 'noise'.
# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------Model with Amp Def ---------------------------------------------------

def Amp_with_Coeff (Extraction_Coeff, Delta_area,):
     Delta_A = Delta_area
     Average_A = 0.2
     one = 1.
     Ext_coeff = Extraction_Coeff

     epsilon = ( one / (Ext_coeff - one))
     Amplitude = (-1 * Delta_A) / (Average_A + epsilon)
     return Amplitude
     # print "\n -----------------------------------------------------------------"
     # print "\n Amplitude with Extraction Coefficient: \n", Amplitude
def make_Amp_x(Datafile):
    def Amp_x (Data_wavelength, Delta_a):
        Intensity = ascii.read(Datafile)
        wavelength = Intensity['col1']
        extraction = Intensity['col2']
        Amp_Ext_Coeff= Amp_with_Coeff(extraction, Delta_a)
        Inter_Amp = np.interp(Data_wavelength, wavelength, Amp_Ext_Coeff)
        return Inter_Amp
    return Amp_x
Delta_a = 0.80

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- List of directories needed for Amp Model------------------------------------

# File that opens the needed intensity files
file_dir_Int_models = "/Users/melaniapena/Rsrch/code/2017_fall_semester/data/Int_Amp_Dec_21_17/Intensities(DNT)"

# this is a nonexistent directory. However, we will make an actual directory using 'os.path.exists('')'
save_dir = "/Users/melaniapena/Rsrch/code/2017_fall_semester/data/Int_Amp_Dec_21_17/(pdf)_plotted_model_amplitudes/"

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------Amplitude Comparison with HST/Spitzer and model of Intensities--------------------------

# This creates a new directory folder. The if-else statement creates a new directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# creates a for loop and looks through the address (finds all the files in a directory)
# only if these files are in ascii format
for file in os.listdir(file_dir_Int_models):
    # joins the file name and rest of directory back together.
    full_file_path = os.path.join(file_dir_Int_models, file)
    # reads the directory file.
    Intensity_File = ascii.read(full_file_path)
    Wavelength_File = Intensity_File['col1']
    Extraction_File = Intensity_File['col2']

    # guess parameter is Delta_a = 0.8. located under 'Model with Amp Def'
    guess_params_a = Delta_a

    # produces the amplitude of the given Intensity data
    best_fit, pcov = curve_fit(make_Amp_x(full_file_path), Wavelengths_of_HST_SPTZ, Amp_of_HST_SPTZ, p0=guess_params_a, bounds=(-0.2, 0.8))
    model_amp_atdata = make_Amp_x(full_file_path)(Wavelengths_of_HST_SPTZ,best_fit)
    model_amp = make_Amp_x(full_file_path)(Wavelength_File,best_fit)

    yerr_vector = [np.median(J_flux_err_med_HST), np.median(W_flux_err_med_HST), np.median(H_flux_err_med_HST), np.median(clip_flux_err_SPTZ)] / (np.sqrt(5))

    # plots both the model amplitude along with the error bars of the HST and Spitzer data.
    plt.plot(Wavelength_File,model_amp,label='Model Amp.')
    plt.errorbar(Wavelengths_of_HST_SPTZ, Amp_of_HST_SPTZ, yerr=yerr_vector, marker='o', linestyle='None',label='HST/SPTZ Amp.')

    #changes the file with the next file
    name_of_file = os.path.splitext(os.path.basename(file))[0]

    # computes chi squared and plots onto graph
    chi_squared = chi_squared_GOF(model_amp_atdata,Amp_of_HST_SPTZ,yerr_vector)
    reduced_chi = round(chi_squared, 3)
    chi_squared_to_figure = "Chi-Squared: " + str(reduced_chi)
    print(name_of_file),('chi-squared: '),(reduced_chi)
    text_title_2 = "Amp Model Comparison with SPTZ/HST Amp"
    plt.title(text_title_2)
    plt.xlabel('Wavelength', fontsize=14)  # labeling Figure
    plt.ylabel('Amplitude', fontsize=14)
    l2 = plt.legend(loc=1)
    plt.figtext(0.02,0.02,chi_squared_to_figure)

    # saves and creates a new file in the new folder.
    plt.savefig(save_dir + name_of_file + ".pdf",dpi=600)
    # plt.grid()
    # plt.show()
    plt.clf()