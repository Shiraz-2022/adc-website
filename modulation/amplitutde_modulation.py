import matplotlib.pyplot as plt
import numpy as np
import math
from .util import *



def round_to_nearest_multiple(number):
        length = len(str(number))
        base = 10 ** (length - 1)
        return base * round(number / base)
#function for ploting amplitude modulation graph

def envelope_detector(modulated_wave):
    envelope = np.abs(modulated_wave)  # Take the absolute value to extract the envelope
    return envelope

def low_pass_filter(envelope, cutoff_frequency, sampling_frequency):
    # Use a simple low-pass filter, such as a moving average, to remove high-frequency components
    # You can also use more advanced filter designs if needed.
    window_size = int(sampling_frequency / cutoff_frequency)
    filter_response = np.ones(window_size) / window_size
    demodulated_signal = np.convolve(envelope, filter_response, mode='same')
    return demodulated_signal    

def AM_main_graph(inputs):
    graphs = [] # created an expty array graphs
    Am,Ac,fm,fc,message_signal = inputs.values() #transfered input values to these variables
    condition = "line" # scattered plotting(dots)


    fm = round_to_nearest_multiple(fm)
    fc = round_to_nearest_multiple(fc)

    x_carrier = create_domain_AM() #calls craet domain function from util with input fc which creates an np linspace
    x_message = create_domain_AM() #calls craet domain function from util with input fm which creates an np linspace
    x_modulated = create_domain_AM() #domain for modulated signal is used based on who has the lesser samples
    carrier = Ac*np.cos(2*np.pi*fc*x_carrier)

   
    if(message_signal=="sin"): # if message signal is sine
        message = Am*np.sin(2*np.pi*fm*x_message) # generate message signal based on amplitude given
    elif message_signal=='cos':
        message = Am*np.cos(2*np.pi*fm*x_message)
    elif message_signal=='tri':
        message = triangular(fm, Am, x_message)

    modulated_wave = (1 + message / Ac) * carrier
    demodulated_wave = modulated_wave * carrier
    # envelope = envelope_detector(modulated_wave)
    # demodulated_wave = low_pass_filter(envelope, 2*fm, 2 *fc)
    demodulated_wave = envelope_detector(demodulated_wave)


    
        
    a = plot_graph(condition = condition, x = x_message, y = message, title = "Message Signal",color='y') # plot graph using plot graph function in util
    b = plot_graph(condition = condition, x = x_carrier, y = carrier, title = "Carrier Signal",color='g')
    c = plot_graph(condition = condition, x = x_modulated, y = modulated_wave, title = "Modulated wave",color='r')
    d = plot_graph(condition = condition, x = x_message, y = demodulated_wave, title="demodulated wave")

    return [a,b,c,d]


def AM_double_sideband_modulation(inputs):
    
    Am,Ac,fm,fc,message_signal = inputs.values()
    condition = "line"

    x_carrier = create_domain_AM()
    x_message = create_domain_AM()
    x_modulated = create_domain_AM()
    
    fm = round_to_nearest_multiple(fm)
    fc = round_to_nearest_multiple(fc)

    carrier = Ac*np.cos(2*np.pi*fc*x_carrier)
   


    if message_signal=="sin":
        message = Am*np.sin(2*np.pi*fm*x_message)
    elif message_signal=='tri':
        message = triangular(fm, Am, x_message)
    elif message_signal=='cos':   
        message = Am*np.cos(2*np.pi*fm*x_message)



    modulated_wave = carrier * message
    demodulated_wave = modulated_wave * carrier
    

    a = plot_graph(condition = condition, x = x_message, y = message, title = "Message Signal", color = 'y')
    b = plot_graph(condition = condition, x = x_carrier, y = carrier, title = "Carrier Signal", color = 'g')
    c = plot_graph(condition = condition, x = x_modulated, y = modulated_wave, title = "Modulated wave", color ='r')
    d = plot_graph(condition = condition, x = x_message, y = demodulated_wave, title="demodulated wave", color = 'm')

    return [a,b,c,d]



def AM_ssb_modulation(inputs):
    Am,Ac,fm,fc,message_signal = inputs.values()
    condition = "line"
    x_carrier = create_domain_AM()
    x_message = create_domain_AM()
    x_modulated = create_domain_AM() #x_carrier if(len(x_carrier)<len(x_message)) else x_message    
    
    fm = round_to_nearest_multiple(fm)
    fc = round_to_nearest_multiple(fc)
    carrier = Ac*np.cos(2*np.pi*fc*x_carrier)


    if message_signal=="sin":
        message = Am*np.sin(2*np.pi*fm*x_message)
    elif message_signal=="cos":
        message = Am*np.cos(2*np.pi*fm*x_message)
    elif message_signal =="tri":
        message = triangular(fm, Am, x_message)

    modulated_positive = message * carrier
    modulated_negative = -message * carrier

    demodulated_wave = (modulated_positive-modulated_negative)*carrier

    
    a = plot_graph(condition = condition, x = x_message, y = message,color='g', title = "Message Signal")
    b = plot_graph(condition = condition, x = x_carrier, y = carrier,color='m', title = "Carrier Signal")
    c = plot_graph(condition = condition, x = x_modulated, y = modulated_positive, color='r', title = "Modulated wave 1",text="upper Sideband")
    d = plot_graph(condition = condition, x = x_modulated, y = modulated_negative, color='b', title = "Modulated wave 2",text="lower Sideband")
    e = plot_graph(condition = condition, x = x_message, y=demodulated_wave,color='r', title="demodulated wave")
    
    return [a,b,c,d,e]

def AM_QAM(inputs):
    Am,Ac,fm,fc,message_signal,message_signal_2 = inputs.values()
    condition="line"
    x_carrier = create_domain_AM()
    x_message = create_domain_AM()
    x_modulated = create_domain_AM() #x_carrier if(len(x_carrier)<len(x_message)) else x_message

    fm = round_to_nearest_multiple(fm)
    fc = round_to_nearest_multiple(fc)

    c1 = Ac*np.cos(2*np.pi*fc*x_carrier) #carrier 1
    c2 = Ac*np.sin(2*np.pi*fc*x_carrier) #carrier 2

    if message_signal=="sin":
        m1 = Am*np.sin(2*np.pi*fm*x_message)
    elif message_signal=="cos":
        m1 = Am*np.cos(2*np.pi*fm*x_message)
    elif message_signal=="tri":
        m1 = triangular(fm, Am, x_message)
    
    if message_signal_2 == "sin":
        m2 = Am*np.sin(2*np.pi*fm*x_message)
    elif message_signal_2 == "cos":
        m2 = Am*np.cos(2*np.pi*fm*x_message)
    elif message_signal_2 == "tri":
        m1 = triangular(x_message, Am)


    modulated_wave_1 = c1 * m1
    modulated_wave_2 = c2 * m2  

    demodulated_wave_1 = modulated_wave_1 * np.cos(2 * np.pi * fc * x_modulated)
    demodulated_wave_2 = modulated_wave_2 * np.sin(2 * np.pi * fc * x_modulated)  
    
    modulated_wave = modulated_wave_1 + modulated_wave_2

    a = plot_graph(condition = condition,x = x_message, y = m1,color='b', title = "Message Signal-1")
    b = plot_graph(condition = condition,x = x_message, y = m2,color='g', title = "Message Signal-2")
    c = plot_graph(condition = condition,x = x_carrier, y = c1,color='m', title = "Carrier Signal-1")
    d = plot_graph(condition = condition,x = x_carrier, y = c2,color='y', title = "Carrier Signal-2")
    e = plot_graph(condition = condition,x = x_modulated, y = modulated_wave_1,color='r', title = "Modulated wave - 1")
    f = plot_graph(condition = condition,x = x_modulated, y = modulated_wave_2,color='r', title = "Modulated wave - 2")
    g = plot_graph(condition = condition,x = x_message, y=demodulated_wave_1,color='r', title="demodulated wave - 1")
    h = plot_graph(condition = condition,x = x_message, y=demodulated_wave_2,color='c', title="demodulated wave - 2")
    
    return [a,b,c,d,e,f,g,h]