# ------- BASK - Binary Amplitude Shift Keying ---------
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO


import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from .util import *

def countNoOfDigits(f):
    count = 0
    while int(f) != 0: 
        count = count+1
        f=f/10
    return count    

    
def countSpace(noOfDigits):
    space = 0
    while noOfDigits != 1: 
        space = space * 10 + 1
        noOfDigits = noOfDigits-1  
    return space   

def round_to_nearest_multiple(number):
        length = len(str(number))
        base = 10 ** (length - 1)
        return base * round(number / base)


def BASK(Tb, fc,Ac1,Ac2, inputBinarySeq):

    fc = round_to_nearest_multiple(fc)
    condition = 'line'
    m = inputBinarySeq.reshape(-1, 1)
    N = len(m) # length of binary sequence
    
    x_carrier = create_domain_AM()

    t = np.arange(0, N * Tb, Tb / 100) 
    A = np.sqrt(2 / Tb) 
    t1 = 0
    t2 = Tb

    bit = np.array([])
    
    for n in range(N):
        if m[n] == 1:
            se = np.ones(100)
        else:
            se = np.zeros(100)
        bit = np.concatenate((bit, se)) 
   
    fDigits = countNoOfDigits(fc)
    space = countSpace(fDigits) * 9    

    t2 = np.arange(Tb / 99, Tb + Tb / 99, Tb / space)

    message = np.array([])
    for i in range(N):
        if m[i] == 1:
            y = Ac1 * np.cos(2 * np.pi * fc * t2)
        else:
            y = Ac2 * np.cos(2 * np.pi * fc * t2)
        message = np.concatenate((message, y))


    #plotting message signal

    plt.subplot(3, 1, 1)
    plt.plot(t, bit, "b", linewidth=2.5)
    plt.grid(True)
    plt.axis([0, Tb * N, -1, 2])
    plt.ylabel("Amplitude (V)")
    plt.xlabel("Time (s)")
    plt.title("Message signal")
    plt.grid(True)

    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    msg = data.getvalue().hex()
    plt.figure()


    c1 = Ac1 * np.cos(2 * np.pi * fc * x_carrier)
    c2 = Ac2 * np.cos(2 * np.pi * fc * x_carrier)

    carrier1 = plot_graph(condition = condition, x = x_carrier, y = c1, title = "Carrier Signal",color='g')
    carrier2 = plot_graph(condition = condition, x = x_carrier, y = c2, title = "Carrier Signal",color='g')

    t3 = np.arange(Tb / 99, Tb * N + Tb / 99,Tb / space)
    plt.subplot(3, 1, 2)
    if Ac1 > Ac2 :
        plt.axis([0, Tb * N, -Ac1 - 5, Ac1 + 5])
    else:  
        plt.axis([0, Tb * N, -Ac2 - 5, Ac2 + 5])  
    plt.plot(t3, message, "r")
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.title("Modulated Wave")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    mod = data.getvalue().hex()
    plt.figure()

    plt.close('all')  # Close all plots

    # return [msg_mod, carrier]
    return [msg,carrier1,carrier2,mod]



# ------- BFSK - Binary Frequency Shift Keying ----------


def BFSK(Tb,Ac, fc1, fc2, inputBinarySeq):
    # Binary Information
    x = inputBinarySeq.reshape(-1, 1) # By using -1, NumPy will automatically calculate the appropriate number of rows and 1 is no fo columns
  
    bp = Tb #but period
    condition = "line"
    fc1 = round_to_nearest_multiple(fc1)
    fc2 = round_to_nearest_multiple(fc2)
    # Representation of transmitting binary information as digital signal
    bit = np.array([])
    
    for n in range(len(x)):
        if x[n] == 1:
            se = np.ones(100)
        else:
            se = np.zeros(100)
        bit = np.concatenate((bit, se))

    t1 = np.arange(0, len(x) * bp, bp / 100)

 

    # Binary-FSK modulation
    # A = np.sqrt(2 / Tb)  # Amplitude of carrier signal
    br = 1 / bp  # bit rate
    f1 = br * fc1  # carrier frequency for information as 1
    f2 = br * fc2  # carrier frequency for information as 0

    if f2>f1:
        fDigits = countNoOfDigits(f2)
    else:
        fDigits = countNoOfDigits(f1)    

    space = countSpace(fDigits) * 9    

    t2 = np.arange(bp / 99, bp + bp / 99, bp / space)
    m = np.array([])
    for i in range(len(x)):
        if x[i] == 1:
            y = Ac * np.cos(2 * np.pi * f1 * t2)
        else:
            y = Ac * np.cos(2 * np.pi * f2 * t2)
        m = np.concatenate((m, y))


    #ploting message signal

    plt.subplot(3, 1, 1)
    plt.plot(t1, bit, "b", linewidth=2.5)
    plt.grid(True)
    plt.axis([0, bp * len(x), -1, 2])
    plt.ylabel("Amplitude (V)")
    plt.xlabel("Time (s)")
    plt.title("Message signal")
    plt.grid(True)

    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    msg = data.getvalue().hex()
    plt.figure()



    x_carrier = create_domain_AM()
    c1 = Ac * np.cos(2 * np.pi * fc1 * x_carrier)
    c2 = Ac * np.cos(2 * np.pi * fc2 * x_carrier)

    carrier1 = plot_graph(condition = condition, x = x_carrier, y = c1, title = "Carrier Signal 1",color='g')
    carrier2 = plot_graph(condition = condition, x = x_carrier, y = c2, title = "Carrier Signal 2",color='g')

    # Modulated Signal
    t3 = np.arange(bp / 99, bp * len(x) + bp / 99, bp / space)
    plt.subplot(3, 1, 2)
    plt.axis([0, bp * len(x), -Ac - 5, Ac + 5])
    plt.plot(t3, m, "r")
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.title("Modulated Wave")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    mod = data.getvalue().hex()
    plt.figure()

    return [msg, carrier1, carrier2, mod]


# ------------- BPSK - Binary Phase Shift Keying ---------
def BPSK(Tb,Ac, fc, inputBinarySeq):
    # x = np.array([1, 0, 0, 1, 1, 0, 1])  # Binary Information
    x = inputBinarySeq.reshape(-1, 1)
    condition = 'line'
    # bp = 0.000001  # bit period
    bp = Tb
    fc = round_to_nearest_multiple(fc)
    # Transmitting binary information as digital signal
    bit = np.array([])
    for n in range(len(x)):
        if x[n] == 1:
            se = np.ones(100)
        else:
            se = np.zeros(100)
        bit = np.concatenate([bit, se])

    t1 = np.arange(bp / 100, 100 * len(x) * (bp / 100) + bp / 100, bp / 100)
    plt.subplot(3, 1, 1)
    plt.plot(t1, bit, linewidth=2.5)
    plt.grid(True)
    plt.axis([0, bp * len(x), -1, 2])
    plt.ylabel("Amplitude(Volt)")
    plt.xlabel("Time(sec)")
    plt.title("Message Signal")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    msgSignal = data.getvalue().hex()
    plt.figure()

    # Binary-PSK modulation
    # A = np.sqrt(2 / Tb)  # Amplitude of carrier signal
    br = 1 / bp  # bit rate
    f = br * 2  # carrier frequency

    fDigits = countNoOfDigits(fc)
    space = countSpace(fDigits) * 9   

    t2 = np.arange(bp / 99, bp + bp / 99, bp / space)
    ss = len(t2)
    m = np.array([])
    for i in range(len(x)):
        if x[i] == 1:
            y = Ac * np.cos(2 * np.pi * fc * f * t2)
        else:
            y = Ac * np.cos(2 * np.pi * fc * f * t2 + np.pi)
        m = np.concatenate([m, y])

    x_carrier = create_domain_AM()
    c = Ac * np.cos(2 * np.pi * fc * x_carrier)

    carrier = plot_graph(condition = condition, x = x_carrier, y = c, title = "Carrier Signal",color='g')

    # Plotting the carrier signal
    # plt.subplot(5, 1, 3)
    # plt.plot(t2, y)
    # plt.title("carrier signal")
    # plt.xlabel("t")
    # plt.ylabel("c(t)")
    # plt.grid(True)
    # # Save
    # data = BytesIO()
    # plt.savefig(data, format="png", bbox_inches="tight")
    # data.seek(0)
    # carrier = data.getvalue().hex()
    # plt.figure()

    # Modulated
    t3 = np.arange(bp / 99, bp * len(x) + bp / 99, bp / space)
    plt.subplot(3, 1, 2)
    plt.plot(t3, m, "r")
    plt.axis([0, bp * len(x), -Ac - 5, Ac + 5])
    plt.xlabel("Time(sec)")
    plt.ylabel("Amplitude(Volt)")
    plt.title("Modulated Wave")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    modulatedSignal = data.getvalue().hex()
    plt.figure()

    return [msgSignal, carrier, modulatedSignal]


# ------- QPSK ---------------
def QPSK(Tb,Ac, fc, inputBinarySeq):
    t = np.linspace(0, 1, 100)  # Time
    condition = 'line'

    m = inputBinarySeq.reshape(-1, 1)
    N = len(m)

    bit = np.array([])
    for n in range(N):
        if m[n] == 1:
            se = np.ones(100)
        else:
            se = np.zeros(100)
        bit = np.concatenate([bit, se])


    # t1 = 0
    # t2 = Tb
    t2 = np.arange(Tb / 99, Tb + Tb / 99, Tb / 99)
    s = np.array([])
    for i in range(0, N, 2):
        if m[i] == 0 and m[i+1] == 0:
            y = Ac * np.cos(2 * np.pi * fc * t2 + np.pi/4)
        elif m[i] == 0 and m[i+1] == 1:
            y = Ac * np.cos(2 * np.pi * fc * t2 + 3*(np.pi/4))
        elif m[i] == 1 and m[i+1] == 0:
            y = Ac * np.cos(2 * np.pi * fc * t2 + 5*(np.pi/4))
        elif m[i] == 1 and m[i+1] == 1:
            y = Ac * np.cos(2 * np.pi * fc * t2 + 7*(np.pi/4))
        s = np.concatenate([s, y])


    x_carrier = create_domain_AM()

    c1 = Ac * np.sqrt(2 / Tb) * np.cos(2 * np.pi * fc * x_carrier)  # carrier frequency cosine wave
    c2 = Ac * np.sqrt(2 / Tb) * np.sin(2 * np.pi * fc * x_carrier)  # carrier frequency sine wave

    carrier1 = plot_graph(condition = condition, x = x_carrier, y = c1, title = "Carrier Signal 1",color='g')
    carrier2 = plot_graph(condition = condition, x = x_carrier, y = c2, title = "Carrier Signal 2",color='g')


    t3 = np.arange(Tb / 99, Tb * N + Tb / 99, Tb / 99)
    plt.subplot(3, 1, 2)
    plt.plot(t3, s, "r")
    plt.axis([0, Tb * N, -Ac - 5, Ac + 5])
    plt.xlabel("Time(sec)")
    plt.ylabel("Amplitude(Volt)")
    plt.title("Modulated Wave")
    plt.grid(True)

    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    modSignal = data.getvalue().hex()
    plt.figure()


    
    t1 = np.arange(Tb / 100, 100 * N * (Tb / 100) + Tb / 100, Tb / 100)
    plt.subplot(3, 1, 1)
    plt.plot(t1, bit, linewidth=2.5)
    plt.grid(True)
    plt.axis([0, Tb * N, -1, 2])
    plt.ylabel("Amplitude(Volt)")
    plt.xlabel("Time(sec)")
    plt.title("Message Signal")
    plt.grid(True)
    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    msgSignal = data.getvalue().hex()
    plt.figure()

    return [msgSignal, carrier1, carrier2, modSignal]
    
    ## modulation
    # odd_sig = np.zeros((len(m), 100))
    # even_sig = np.zeros((len(m), 100))

    # plt.subplot(3, 1, 2)
    # for i in range(0, len(m) - 1, 2):
    #     t = np.linspace(t1, t2, 100)
    #     if m[i] > 0.5:
    #         m[i] = 1
    #         m_s = np.ones((1, len(t)))
    #     else:
    #         m[i] = 0
    #         m_s = (-1) * np.ones((1, len(t)))

    #     odd_sig[i, :] = c1 * m_s

    #     if m[i + 1] > 0.5:
    #         m[i + 1] = 1
    #         m_s = np.ones((1, len(t)))
    #     else:
    #         m[i + 1] = 0
    #         m_s = (-1) * np.ones((1, len(t)))

    #     even_sig[i, :] = c2 * m_s

    #     qpsk = odd_sig + even_sig  # modulated wave = oddbits + evenbits

    #     plt.plot(t, qpsk[i, :])
    #     t1 = t1 + (Tb + 0.01)
    #     t2 = t2 + (Tb + 0.01)


    # plt.title("Modulated Wave")
    # plt.grid(True)
    # # Save
    # data = BytesIO()
    # plt.savefig(data, format="png", bbox_inches="tight")
    # data.seek(0)
    # modSignal = data.getvalue().hex()
    # plt.figure()


    # # Message Signal
    # plt.figure()
    # plt.subplot(3, 1, 2)
    # plt.stem(range(len(m)), m, use_line_collection=True)
    # plt.ylabel("Binary value")
    # plt.title("Message signal")
    # plt.grid(True)
    # # Save
    # data = BytesIO()
    # plt.savefig(data, format="png", bbox_inches="tight")
    # data.seek(0)
    # msgSignal = data.getvalue().hex()
    # plt.figure()

    # plt.subplot(3, 1, 2)
    # plt.plot(t, c1)
    # plt.xlabel("Time (Number of samples)")
    # plt.ylabel("Cos Wave")
    # plt.title("Carrier Wave 1 (Cosine)")
    # plt.grid(True)
    # # Save
    # data = BytesIO()
    # plt.savefig(data, format="png", bbox_inches="tight")
    # data.seek(0)
    # carrier1 = data.getvalue().hex()
    # plt.figure()

    # plt.subplot(3, 1, 2)
    # plt.plot(t, c2)
    # plt.xlabel("Time (Number of samples)")
    # plt.ylabel("Sine Wave")
    # plt.title("Carrier Wave 2 (Sine)")
    # plt.grid(True)
    # # Save
    # data = BytesIO()
    # plt.savefig(data, format="png", bbox_inches="tight")
    # data.seek(0)
    # carrier2 = data.getvalue().hex()
    # plt.figure()







# ------- GMSK ---------------
def gaussianLPF(BT, Tb, L, k):
    """
    Generate filter coefficients of Gaussian low pass filter (used in gmsk_mod)
    Parameters:
        BT : BT product - Bandwidth x bit period
        Tb : bit period
        L : oversampling factor (number of samples per bit)
        k : span length of the pulse (bit interval)
    Returns:
        h_norm : normalized filter coefficients of Gaussian LPF
    """
    B = BT / Tb  # bandwidth of the filter
    # truncated time limits for the filter
    t = np.arange(start=-k * Tb, stop=k * Tb + Tb / L, step=Tb / L)
    h = (
        B
        * np.sqrt(2 * np.pi / (np.log(2)))
        * np.exp(-2 * (t * np.pi * B) ** 2 / (np.log(2)))
    )
    h_norm = h / np.sum(h)
    return h_norm


def GMSK(a, fc, L, BT):
    """
    Function to modulate a binary stream using GMSK modulation
    Parameters:
        a : input binary data stream (0's and 1's) to modulate (string)
        fc : RF carrier frequency in Hertz
        L : oversampling factor
        BT : BT product (bandwidth x bit period) for GMSK
    Returns:
        (s_t,s_complex) : tuple containing the following variables
            s_t : GMSK modulated signal with carrier s(t)
            s_complex : baseband GMSK signal (I+jQ)
    """
    from scipy.signal import upfirdn, lfilter

    # Change String of DataStream to numpy array
    a = np.array(list(a), dtype=int)

    fs = L * fc
    Ts = 1 / fs
    Tb = L * Ts
    # derived waveform timing parameters
    c_t = upfirdn(h=[1] * L, x=2 * a - 1, up=L)  # NRZ pulse train c(t)

    k = 1  # truncation length for Gaussian LPF
    h_t = gaussianLPF(BT, Tb, L, k)  # Gaussian LPF with BT=0.25
    b_t = np.convolve(h_t, c_t, "full")  # convolve c(t) with Gaussian LPF to get b(t)
    bnorm_t = b_t / max(abs(b_t))  # normalize the output of Gaussian LPF to +/-1

    h = 0.5
    # integrate to get phase information
    phi_t = lfilter(b=[1], a=[1, -1], x=bnorm_t * Ts) * h * np.pi / Tb

    I = np.cos(phi_t)
    Q = np.sin(phi_t)  # cross-correlated baseband I/Q signals

    s_complex = I - 1j * Q  # complex baseband representation
    t = Ts * np.arange(start=0, stop=len(I))  # time base for RF carrier
    sI_t = I * np.cos(2 * np.pi * fc * t)
    sQ_t = Q * np.sin(2 * np.pi * fc * t)
    s_t = sI_t - sQ_t  # s(t) - GMSK with RF carrier

    fig, axs = plt.subplots(2, 4, figsize=(15, 5))

    # Adjust vertical spacing between subplots
    fig.subplots_adjust(hspace=0.4)

    axs[0, 0].plot(np.arange(0, len(c_t)) * Ts, c_t)
    axs[0, 0].set_title("c(t)")
    axs[0, 0].set_xlim(0, 40 * Tb)

    axs[0, 1].plot(np.arange(-k * Tb, k * Tb + Ts, Ts), h_t)
    axs[0, 1].set_title("$h(t): BT_b$=" + str(BT))

    axs[0, 2].plot(t, I, "--")
    axs[0, 2].plot(t, sI_t, "r")
    axs[0, 2].set_title("$I(t)cos(2 \pi f_c t)$")
    axs[0, 2].set_xlim(0, 10 * Tb)

    axs[0, 3].plot(t, Q, "--")
    axs[0, 3].plot(t, sQ_t, "r")
    axs[0, 3].set_title("$Q(t)sin(2 \pi f_c t)$")
    axs[0, 3].set_xlim(0, 10 * Tb)

    axs[1, 0].plot(np.arange(0, len(bnorm_t)) * Ts, bnorm_t)
    axs[1, 0].set_title("b(t)")
    axs[1, 0].set_xlim(0, 40 * Tb)

    axs[1, 1].plot(np.arange(0, len(phi_t)) * Ts, phi_t)
    axs[1, 1].set_title("$\phi(t)$")

    axs[1, 2].plot(t, s_t)
    axs[1, 2].set_title("s(t)")
    axs[1, 2].set_xlim(0, 20 * Tb)
    axs[1, 3].plot(I, Q)
    axs[1, 3].set_title("constellation")

    # fig.show()

    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    All_plots = data.getvalue().hex()
    plt.figure()

    return [All_plots]


# -------DPSK ----------
def DPSK(fm, Am, phi_m, fc, Ac, phi_c):
    # Define the time domain
    t_start = 0
    t_stop = 1
    t_step = 0.00001
    t = np.arange(t_start, t_stop, t_step)

    # # Get user input values
    # fm = float(input("Enter the frequency of the message signal (in Hz): "))
    # Am = float(input("Enter the amplitude of the message signal: "))
    # phi_m = float(input("Enter the phase of the message signal (in radians): "))
    # fc = float(input("Enter the frequency of the carrier signal (in Hz): "))
    # Ac = float(input("Enter the amplitude of the carrier signal: "))
    # phi_c = float(input("Enter the phase of the carrier signal (in radians): "))

    # fm = 10
    # Am = 1
    # phi_m = 0
    # fc = 100
    # Ac = 1
    # phi_c = 90

    m = Am * np.cos(2 * np.pi * fm * t + phi_m)

    c = Ac * np.cos(2 * np.pi * fc * t + phi_c)

    delta_phi = np.pi / 2
    s = Ac * np.cos(2 * np.pi * fc * t + phi_c + delta_phi * (m > 0))

    c_demod = Ac * np.cos(2 * np.pi * fc * t)

    r = s * c_demod

    fig, axs = plt.subplots(5, 1, figsize=(20, 20))

    axs[0].plot(t, m)
    axs[0].set_title("Message signal")

    axs[1].plot(t, c)
    axs[1].set_title("Carrier signal")

    axs[2].plot(t, s)
    axs[2].set_title("DPSK-modulated signal")

    axs[3].plot(t, c_demod)
    axs[3].set_title("Carrier signal for demodulation")

    axs[4].plot(t, r)
    axs[4].set_title("DPSK-demodulated signal")

    for ax in axs:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    plt.tight_layout()

    # Save
    data = BytesIO()
    plt.savefig(data, format="png", bbox_inches="tight")
    data.seek(0)
    All_plots = data.getvalue().hex()
    plt.figure()

    return [All_plots]
