import matplotlib.pyplot as plt
import numpy as np
from scipy import signal  
from .util import *


def round_to_nearest_multiple(number):
        length = len(str(number))
        base = 10 ** (length - 1)
        return base * round(number / base)

def FM_MAIN(inputs):

    Am,Ac,fm,fc,message_signal,k = inputs.values()

    fm = round_to_nearest_multiple(fm)
    fc = round_to_nearest_multiple(fc)

    x_carrier = create_domain_AM()
    x_message = create_domain_AM()

    carrier = Ac*np.cos(2*np.pi*fc*x_carrier)
    

    if(message_signal=="sin"):
       # modulated_wave= Ac*(np.cos((2*np.pi*fc*x))+((K*Am/fm)*np.sin(2*np.pi*fm*x)))
        message = Am*np.sin(2*np.pi*fm*x_message )#message signal
    elif(message_signal=="cos"):
       # modulated_wave= Ac*(np.cos(2*np.pi*fc*x)+((K*Am/fm)*np.cos(2*np.pi*fm*x)))
        message = Am*np.cos(2*np.pi*fm*x_message )
    elif(message_signal=="tri"):    
        message = triangular(fm, Am, x_message)

    modulated_wave = Ac * np.cos(2 * np.pi * fc * x_message + k * np.sin(2 * np.pi * fm * x_message)) #Ac * np.cos(2 * np.pi * (fc + k*message) * x_message )
    
    #demodulated_wave = signal.detrend(signal.hilbert(modulated_wave).imag)  
    #demodulated_wave = np.gradient(instantaneous_phase) / (2 * np.pi * k * x_message)
        
    a = plot_graph(x = x_message , y = message,color="red", title = "Message Signal")
    b = plot_graph(x = x_message , y = carrier,color="blue", title = "Carrier Signal")
    c = plot_graph(x = x_message , y = modulated_wave,color="green", title = "Modulated wave")
    #d = plot_graph(x = x_message , y = demodulated_wave,color="red", title = "Demodulated wave")
    return [a,b,c]
    
    
def PHASE_MAIN(inputs):
    Am,Ac,fm,fc,message_signal,k = inputs.values()

    fm = round_to_nearest_multiple(fm)
    fc = round_to_nearest_multiple(fc)

    x_carrier = create_domain_AM()
    x_message = create_domain_AM()
    
    if(message_signal=="sin"):
       # modulated_wave= Ac*(np.cos((2*np.pi*fc*x))+((K*Am/fm)*np.sin(2*np.pi*fm*x)))
        message = Am*np.sin(2*np.pi*fm*x_message )#message signal
    elif(message_signal=="cos"):
       # modulated_wave= Ac*(np.cos(2*np.pi*fc*x)+((K*Am/fm)*np.cos(2*np.pi*fm*x)))
        message = Am*np.cos(2*np.pi*fm*x_message )
    elif(message_signal=="tri"):    
        message = triangular(fm, Am, x_message)    

    sampling_rate = 1000000  # Sampling rate (samples per second)
    t = np.linspace(0, 1, int(sampling_rate), endpoint=False)


    message_int = message.astype(int)
    # for val in message:
    #     if message[val] >= 0:
    #         phase = k * np.sin(2 * np.pi * 5 * x_message)
    #     else:
    #         phase = -k * np.sin(2 * np.pi * 5 * x_message)

    # pm_modulated_signal = np.sin(2 * np.pi * fc * x_message + phase) 

    # k = k*10
    carrier = Ac*np.cos(2*np.pi*fc*t)
    modulated_wave = Ac * np.cos(2 * np.pi * fc * t + k *  message)
    #demodulated_wave = np.gradient(np.unwrap(np.angle(modulated_wave))) / (2 * np.pi * k)
        
    a = plot_graph(x = x_message, y = message, title = "Message Signal",color="red")
    b = plot_graph(x = x_carrier, y = carrier, title = "Carrier Signal",color="green") 
    c = plot_graph(x = x_message, y = modulated_wave, title = "Modulated wave",color="blue") 
    #d = plot_graph(x = x_message, y = demodulated_wave, title = "demodulated wave",color="green")      
    return [a,b,c]

