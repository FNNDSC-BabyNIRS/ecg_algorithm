#This try is implemented to avoid a failure when running locally
try: 
    import nirscloud_util_meta
    import nirscloud_util_hdfs
except: 
    pass
import logging
import pandas as pd
import numpy as np
import plotly
import pyspark
import sys
from plotly.subplots import make_subplots
import math 
import plotly.graph_objects as go
import sys
import matplotlib.pyplot as plt
from numpy import matrix
from scipy.signal import find_peaks
import datetime
from scipy.signal import find_peaks,butter, lfilter, lfilter_zi, convolve,resample, correlate, iirnotch, filtfilt, stft
from scipy import signal


def time_f(time_array):

    mins = ((time_array/1000)/60).astype(int);
    sec = ((time_array/1000)%60).astype(int);
    mili = ((time_array - (mins*60*1000 + sec*1000))*100).astype(int)/100 #*100.astype(int)/100 is to round to two decimals

    return (np.array([str(minutes)+':'+str(seconds)+':'+str(milis) for minutes, seconds, milis in zip(mins,sec,mili)]))

# raw_sample is for the sample extracted that includes the peaks from the cluster already


# This moving average function which serves as a difference equation is a central moving average which means that you are 
# averaging a point taking into account both sides neighbors. So at last you will need to cut either (window_length-1)/2)
# or (window_length-2)/2) depending wether you are having an even or odd number in the window length

def moving_average(window_length, y):
    
    x=y
    M = window_length - 1
    z = np.zeros(M+1)
    x_modified = np.concatenate((z,x[0:len(x) - (M+1)]))
    y1= (1/(M+1)) * (x[0]-x_modified[0])
    yn = [y1]

    for n in range (1,len(x)):
        y_value = ((1/(M+1)) * (x[n]-x_modified[n])) + yn[n-1]
        yn = np.append(yn,y_value)

    if window_length% 2 == 1: 
        cutting_points = int((window_length-1)/2) 
        yn = yn[cutting_points:len(yn)]
        
    else: 
        cutting_points = int((window_length-2)/2)
        yn = yn[cutting_points:len(yn)]
        
    return yn, cutting_points


def average_filter(window_length,signal):

    m2 = window_length - 1

    u = np.heaviside(np.arange(0,len(signal)),1)
    u_sub = np.concatenate([np.zeros(window_length),np.heaviside(np.arange(0,len(signal)-(window_length)),1)])
    h_n_s= (1/(window_length)) * (u - u_sub)

    mov_av_s = np.convolve (h_n_s,signal)
    mov_av_s_sh = mov_av_s[int(m2/2):len(signal)+int(m2/2)]
    
    return mov_av_s_sh




            
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y




matrices={}
three_spline_m = np.array([[4, 1, 0, 0, 0, 0]])
four_spline_m =  np.array([[4, 1, 0, 0, 0, 0, 0, 0]])

two_spline_m =  np.array([[4, 1, 0, 0], [1, 4, 1, 0], [0, 1 , 4 , 1], [0, 0, 1, 4]])

new_row_3 = np.array([1, 4, 1, 0, 0, 0,])
new_row_4 = np.array([1, 4, 1, 0, 0, 0, 0, 0])

for l in range(0,(6-2)):
    three_spline_m = np.vstack([three_spline_m, np.roll(new_row_3, l)])
three_spline_m = np.vstack([three_spline_m, np.array([[0, 0, 0, 0, 1, 4]])])

for u in range(0,(8-2)):
    four_spline_m = np.vstack([four_spline_m, np.roll(new_row_4, u)])
four_spline_m = np.vstack([four_spline_m, np.array([[0, 0, 0, 0, 0, 0, 1, 4]])])

inverse_two_spline_m = np.linalg.inv(two_spline_m); inverse_three_spline_m = np.linalg.inv(three_spline_m); inverse_four_spline_m = np.linalg.inv(four_spline_m)
matrices['2_splines'] = inverse_two_spline_m; matrices['3_splines'] = inverse_three_spline_m; matrices['4_splines'] = inverse_four_spline_m;


# found_peaks needs to be the index of the peaks in the raw_data_values or raw_data_time
# This function returns the maximum or the interpolated peaks and the interpolated peak itself
# splines is how many splines you want to carry out around the peak, minumum is 1 for it to have  
# one polynomial to the right and one to the left

def interpolation_spline(found_peaks, raw_data_time, raw_data_values, splines, resolution):
    
    res = 1000/resolution
    
    # First identify if we are past the peak or before 
    h = 4;
    max_indices = []
    
    #amount of control points
    amount_of_peaks = len(found_peaks)
    corrected_peaks_dict = {}; interpolated_peak_y = {} ;interpolated_peak_x = {};
    peaks_x = []; peaks_y = []; max_x = []; max_y = [];

    f = np.zeros(3)
    for i in range (0,amount_of_peaks): 

        #print(i)

        index = found_peaks[i]
        point = raw_data_values[index]
        previous_point = raw_data_values[index-1]

        #Finding the maximum
        
        if previous_point < point: 
            while previous_point < point: 
                index = index + 1
                point = raw_data_values[index]
                previous_point = raw_data_values[index-1]

            # When the while isn't valid the maximum is located at index-1
            max_idx = index - 1


        #point will be the point past the maximum 
        
        else: 
            while previous_point > point: 
                index = index - 1
                point = raw_data_values[index]
                previous_point = raw_data_values[index-1]

            #when the while finished or isn't valid the maximum is located at index
            
            max_idx = index
        
        max_indices = np.append(max_indices,max_idx)
        last_p = max_idx + splines + 2
        first_p = max_idx - (splines + 1)

        # r and x include from x_{-1} up to x_{n+3}
        r = raw_data_values[first_p:last_p+1]
        x = raw_data_time[first_p:last_p+1]

        # interpolation 
        
        
        #M_(-1)
        m_1 =  (r[2] - (2*r[1]) + r[0]) / (h**2)
        #M_(n+1)
        m_n_1 =  (r[-1] - (2*r[-2]) + r[-3]) /(h**2)
        ys =[]
        ys = np.append( ys, ( (6/(h**2)) * (r[0] - (2*r[1]) + r[2]) ) + m_1 )
        actual_cp = splines*2
        
        for n in range(1,(actual_cp-2)+1):
            ys = np.append( ys, (6/(h**2)) * (r[n] - (2*r[n+1]) + r[n+2]) )
        
        ys = np.append( ys, (6/(h**2)) * (r[actual_cp-1] - (2*r[actual_cp]) + r[actual_cp+1]) - m_n_1)
        
        
        # Now the define inverted matrices
        
        inverted_matrix = matrices[str(splines)+'_splines']
        
        
        Ms = np.matmul(inverted_matrix, ys)
        b = Ms/2
        #We are appending m_n_1 which is the variable for M_{n+1} because we need it to calculate the c's
        Ms = np.append(Ms,m_n_1)
        a = (Ms[1:actual_cp+1] - Ms[0:actual_cp])/(6*h)
        c = ((r[2:(actual_cp+1)+1] - r[1:actual_cp + 1])/h) - ((h/6)*(Ms[1:actual_cp+1]+(2*Ms[0:actual_cp])))
        d = r[1:actual_cp +1] 
        # Remember the plus one is to include the last element which is the actual_cp index other wise python doesn't touch the last elemment
        
        interpolated_y = []
        
        #Piecewise polynomials
        for k in range (1,(splines*2)): 
            x_int = np.arange(x[k],x[k+1],res)
            y_int = (a[k-1]*((x_int - x[k])**3)) + (b[k-1]*((x_int - x[k])**2)) + (c[k-1]*(x_int-x[k])) + d[k-1]
            interpolated_y = np.concatenate((interpolated_y,y_int))
            
        # We delete every last point so that it doesn't repeat with the first of the next piecewise polynomial
        # So this last block of code is to include the last interpolation with the last point
        l_cp = splines*2
        x_int = np.arange(x[l_cp],x[l_cp+1]+res,res)
        y_int = (a[l_cp-1]*((x_int - x[l_cp])**3)) + (b[l_cp-1]*((x_int - x[l_cp])**2)) + (c[l_cp-1]*(x_int-x[l_cp])) + d[l_cp-1]
        interpolated_y = np.concatenate((interpolated_y, y_int))
        interpolated_x = np.arange(x[1],x[(splines*2)+1]+res ,res)
        
        interpolated_peak_y['peak: '+str(i)] = interpolated_y; interpolated_peak_x['peak: '+str(i)] = interpolated_x; 
        
        # getting the maximum collecting  vallues
        max_index = np.where(max(interpolated_y)==interpolated_y)

        peaks_x = np.append(peaks_x,interpolated_x[max_index][0])
        peaks_y = np.append(peaks_y, interpolated_y[max_index][0])
        
        #Analytical Solution
        
        roots_1 = []; roots_2 = []; s_eval_roots = []; s2_eval_roots =[]
        aj = a[splines-1:splines+1]; bj = b[splines-1:splines+1]; cj = c[splines-1:splines+1]; dj = d[splines-1:splines+1]; xj = x[splines:splines+2]
        
        #Coefficients of general formula 1 is for the first piecewise poolynomial and 2 is for the second piecewise polynomial
        g_a1 = 3*aj[0]; g_b1 = (2*bj[0] - (2*xj[0]*3*aj[0])); g_c1 = (3*aj[0]*(xj[0]**2)) - (2*bj[0]*xj[0]) + cj[0]
        g_a2 = 3*aj[1]; g_b2 = (2*bj[1] - (2*xj[1]*3*aj[1])); g_c2 = (3*aj[1]*(xj[1]**2)) - (2*bj[1]*xj[1]) + cj[1]

        roots_1 = np.append(roots_1, (- g_b1 + np.sqrt((g_b1**2) - (4 * g_a1 * g_c1 )))/(2*g_a1) );
        roots_1 = np.append(roots_1, (- g_b1 - np.sqrt((g_b1**2) - (4 * g_a1 * g_c1 )))/(2*g_a1) );
        s_eval_roots = np.append(s_eval_roots, aj[0]*((roots_1[0]-xj[0])**3) + bj[0]*((roots_1[0]-xj[0])**2) + cj[0]*(roots_1[0]-xj[0]) + dj[0])
        s_eval_roots = np.append(s_eval_roots, aj[0]*((roots_1[1]-xj[0])**3) + bj[0]*((roots_1[1]-xj[0])**2) + cj[0]*(roots_1[1]-xj[0]) + dj[0])
        
        if roots_1[np.argmax(s_eval_roots)] <=xj[1]:
            max_x = np.append(max_x, roots_1[np.argmax(s_eval_roots)])
            max_y = np.append(max_y, s_eval_roots[np.argmax(s_eval_roots)])
            
        else: 
            roots_2 = np.append(roots_2, (- g_b2 + np.sqrt((g_b2**2) - (4 * g_a2 * g_c2 )))/(2*g_a2) );
            roots_2 = np.append(roots_2, (- g_b2 - np.sqrt((g_b2**2) - (4 * g_a2 * g_c2 )))/(2*g_a2) );
            s2_eval_roots = np.append(s2_eval_roots, aj[1]*((roots_2[0]-xj[1])**3) + bj[1]*((roots_2[0]-xj[1])**2) + cj[1]*(roots_2[0]-xj[1]) + dj[1])
            s2_eval_roots = np.append(s2_eval_roots, aj[1]*((roots_2[1]-xj[1])**3) + bj[1]*((roots_2[1]-xj[1])**2) + cj[1]*(roots_2[1]-xj[1]) + dj[1])
            max_x = np.append(max_x, roots_2[np.argmax(s2_eval_roots)])
            max_y = np.append(max_y, s2_eval_roots[np.argmax(s2_eval_roots)])
            
    max_indices = np.asarray(max_indices, dtype = 'int')
        
    #max_x and max_y is for analytical solutions, peaks_x and peaks_y is for maximum by collecting data, and interpolated_peak_x/y is the actual interpolation
    return max_x, max_y, peaks_x, peaks_y, interpolated_peak_x, interpolated_peak_y,max_indices


def first_peaks_height_av(y,segments,fraction_of_a_second):

    s = 0
    max_values = []
    
    f = 250*fraction_of_a_second

    while s < segments: 

        max_values = np.append(max_values,np.max(y[int(f*s):int(f*(s+1))]))
        s = s+1

    average_peak_h = np.sum(max_values)/segments
    
    return average_peak_h, max_values



def dif_eq_window_integration(function,window_length,divide):
    
    pivot_function = function
    
    for i in range(1,window_length):
        pivot_function = np.insert(pivot_function,0,0)
        pivot_function = np.delete(pivot_function,len(pivot_function)-1)
        #print(pivot_function)
        function = function + pivot_function

        if divide: 
            result = (1/N)*function
        else: 
            result = function
            
    return result

# This function returns the indices of the intaken data where the fiducial point is
def find_fiducial_point(y,x,test_segments,segment_duration,window_length):
    fiducial_point = []
    peak = 0
    found_p = False
    mwi_peak = []
    look_for_peaks = True
    counter = 0
    peak_average, first_peaks = first_peaks_height_av(y,test_segments,0.8)

    for i in range(1,len(x)-1):
        if look_for_peaks:    
            f_derivative = (y[i+1]-y[i])/(x[i+1]-x[i])

            if found_p:
                if f_derivative > 0: 
                    found_p = False
                    if (y[i]<(0.9)*peak) : 
                        found_p = True
                        
                if (peak/2>y[i]) :
                    fiducial_point = np.append(fiducial_point,i-window_length)
                    mwi_peak = np.append(mwi_peak,y[i])
                    found_p = False
                    peak = 0
                    look_for_peaks = False
                    if len(fiducial_point)>test_segments:
                        peak_average = np.average(mwi_peak)
                        
            else: 
                b_derivative = (y[i]-y[i-1])/(x[i]-x[i-1])
                if (f_derivative < 0) and (b_derivative > 0) and y[i]>0.5*peak_average: 
                    peak = y[i]
                    found_p = True
                    
        else: 
            counter = counter + 1
            if counter== 45:
                counter = 0 
                look_for_peaks = True
            

    
    fiducial_point = np.asarray(fiducial_point, dtype = 'int')

    return fiducial_point




# This function takes two arrays of peaks, where each value in the array is the index of the peak, and finds 
# if any of them have missing peaks compared to the other array
# Index comparing is when you are comparing the index of two different arrays but you might want to compare actual elements

def find_missing_peaks(a_peaks, b_peaks, index_comparing):
    missing_b_peaks = []; missing_a_peaks = []

    if len(a_peaks)<len(b_peaks):
        #a_peaks is the shortest in length
        a_peaks=np.append(a_peaks,np.zeros(len(b_peaks) - len(a_peaks)))        

    else: 
        #b_peaks is the shortest in length
        b_peaks=np.append(b_peaks,np.zeros(len(a_peaks) - len(b_peaks)))

        
    difference = abs(a_peaks - b_peaks) #Subtraction order don't matter

    a_insertions = 0; b_insertions = 0;

    while (difference>70).any():
        

        #Here we first find in what index is where a mark is misssing

        index = np.min(np.where(difference>70))
        #This defines if either the short pivot or the long pivot array is missing a mark on a peak
        
        # If True then there's a missing peak in the b pivot

        if (a_peaks[index] - b_peaks[index]) < 0:
            

            #This is for handling the last zeros. When you arrive to the ending zeros it means that you will not do 
            # more insertions but rather replace the zeros for the peaks
            
            if a_peaks[index] == 0:
                a_peaks = np.insert(a_peaks,index,b_peaks[index])
                a_peaks = np.delete(a_peaks,-1)
                missing_a_peaks = np.append(missing_a_peaks,index-b_insertions)

            else:
                missing_b_peaks = np.append(missing_b_peaks,index-a_insertions)
                b_peaks = np.insert(b_peaks,index,a_peaks[index])
                b_insertions = b_insertions + 1
                
                
                # If last element is zero you can erease last element, otherwise append a 0 to the other array. 
                if b_peaks[-1] == 0 :
                    b_peaks = np.delete(b_peaks,-1)
                else:
                    a_peaks =  np.append(a_peaks,0)

        #There is a missing peak in the a_peaks
        else:
            
           #This is for handling the last zeros. 
            if b_peaks[index] == 0:

                b_peaks = np.insert(b_peaks,index,a_peaks[index])
                b_peaks = np.delete(b_peaks,-1)
                missing_b_peaks = np.append(missing_b_peaks,index-a_insertions)


            else:
                missing_a_peaks = np.append(missing_a_peaks,index - b_insertions)
                a_peaks = np.insert(a_peaks,index,b_peaks[index])
                a_insertions = a_insertions + 1
                
                if a_peaks[-1] == 0 :
                    a_peaks = np.delete(a_peaks,-1)
                else:
                    b_peaks =  np.append(b_peaks,0)
                
        difference = abs(a_peaks - b_peaks)
    
   
    if index_comparing: 
        missing_a_peaks = np.asarray(missing_a_peaks, dtype = 'int'); missing_b_peaks = np.asarray(missing_b_peaks, dtype = 'int')
    else: 
        missing_a_peaks = np.asarray(missing_a_peaks, dtype = 'float'); missing_b_peaks = np.asarray(missing_b_peaks, dtype = 'float')
        
        
    # The outcome are the indices that are missing in the other array. For example if a is missing the index N in the b array 
    # then the index N is stored in missing_a_peaks
    # missing_a_peaks are indices of the b array, with elements that a doesn't contain
    return missing_a_peaks, missing_b_peaks




# This function takes mainly the amount of windows one can have in the available time and calculate the overlapping mean values

def overlapping_mean(total_time,heart_beat_time, time_window_length, windows): 
    # This block calculates the mean heartrate of the different windows 
    print('total_time: '+str(total_time) + ' mins')
    print('')
    print('windows: '+ str(windows))

    first_array = np.delete(heart_beat_time,0)
    second_array = np.delete(heart_beat_time,-1)

    dif = (first_array - second_array)/1000
    heart_rate = 60/dif

    # k is the index where the cumulative sum of time is greater than half the time interval
    k = 0
    cmltive_time = 0

    #The even_idx and odd_idx arrays store the indices where the windows start and finish.
    even_idx = []
    odd_idx = [0]
    odd_window_mean = []
    even_window_mean = []

    # The total time of the interval will be defined by the amount of windows one chooses, the overlap and the length of the windows. 
    #This first loop is to find the index of the beginning of the second window 
    while cmltive_time < (time_window_length*60)/2 : 
        k = k + 1
        cmltive_time = np.sum(dif[0:k])
    even_idx = np.append(even_idx,k-1)

    leading_time = time_window_length*60

    window_counter = 1
    odd = True 


    #At the end of the while there is still one more window to calculate its mean.
    while window_counter < windows : 
        while cmltive_time < leading_time: 
            k = k + 1 
            cmltive_time = np.sum(dif[0:k])

        # The reason we append the index k-1 is because when the condition cmltive_time < leading_time isn't met you actually want 
        # the index from the sample before where the condition was actually met, which is k-1. 

        # The 'dif' array is actually the heartrate array. So what we need to do is to consider all of the heartbeats in that array for the established windows. 

        if odd:
            odd = False 
            odd_idx = np.append(odd_idx,k-1)
            odd_idx = np.asarray(odd_idx, dtype = 'int')
            odd_window_mean = np.append(odd_window_mean ,np.mean(heart_rate[odd_idx[-2]:odd_idx[-1]]))

        else: 
            odd = True
            even_idx = np.append(even_idx,k-1)
            even_idx = np.asarray(even_idx, dtype = 'int')
            even_window_mean = np.append(even_window_mean, np.mean(heart_rate[even_idx[-2]:even_idx[-1]]))

        leading_time = (time_window_length + (1-overlap_window)*(time_window_length)*(window_counter))*60
        window_counter = window_counter + 1

    #In this last part we include the mean of the last window. 

    if odd:
        odd_idx=np.append(odd_idx,len(dif))
        odd_window_mean = np.append(odd_window_mean ,np.mean(heart_rate[odd_idx[-2]:odd_idx[-1]]))

    else: 
        even_idx=np.append(even_idx,len(dif))
        even_window_mean = np.append(even_window_mean ,np.mean(heart_rate[even_idx[-2]:even_idx[-1]]))

    overlap_mean_values=odd_window_mean
    adding = 1

    #Joinning the mean odd and even window arrays to a single one in order. 
    for j in range(0, len(even_window_mean)):
        overlap_mean_values = np.insert(overlap_mean_values,j+adding,even_window_mean[j])
        adding = adding + 1
    
    return overlap_mean_values
