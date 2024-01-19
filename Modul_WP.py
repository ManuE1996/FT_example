import matplotlib.pyplot as plt
import numpy as np

def V_Box(x,B_pos,B_height):
    V_ext = np.zeros_like(x)
    for key,element in enumerate(x):
        if element > B_pos[0] and element < B_pos[1]:
            V_ext[key] = 0
        else:
            V_ext[key] = B_height
    return V_ext

def V_Hosz(x,w,k):
    V_ext=w*x**2-k
    return V_ext

def int(x,dx):
    res1 = np.sum(x*dx,axis=0)
    return res1

def dif(x,dx,WD):
    N=len(x)
    main = [0    ,-2]
    Up   = [-0.5 ,1]
    Down = [0.5  ,1]
    diag    = np.ones(N)   * main[WD-1]
    offup   = np.ones(N-1) * Up[WD-1]
    offdown = np.ones(N-1) * Down[WD-1]
    matrix = (np.diag(diag)+np.diag(offup,-1)+np.diag(offdown,+1))/dx**WD
    return matrix

def Build_WF(c_n,psi,E,t):
    Psi = np.zeros(len(psi),dtype=complex)

    for k in range(len(E)):
        Psi += c_n[k]*psi[k,:]*np.exp(-1j*E[k]*t)

    return Psi


def DFT_new(func,time_start,time_stop,samp_rate):
    Nop       = len(func)                                                       #Number of Points from input vector
    T_add     = Nop-(time_stop-time_start)*samp_rate
    k_in      = np.arange(time_start*samp_rate,time_stop*samp_rate+T_add,1)     #Indize of the inner summation
    k_out     = np.arange(Nop).reshape((Nop,1))                                 #Indize of the outer summation
    exp_part  = np.exp(-2j*np.pi*k_out*k_in/Nop)
    result    = np.dot(exp_part,func)
    return result

def freq_FD(x,samp_rate):
    N=len(x)
    n=np.arange(N)
    result=n*samp_rate/N
    return result

def Peak_detection(x_array,y,Peak_T):
    matrix=np.empty((0,3))
    y_max=np.max(abs(y))
    LOP=[]
    for k, x in enumerate(x_array):
        if abs(y[k]) >= abs(y_max)*Peak_T:
            if abs(y[k]) >= abs(y[k-1]) and abs(y[k]) >= abs(y[k+1]):
                Peak = np.array([abs(x),abs(y[k]),np.angle(y[k])*180/np.pi])
                LOP.append(Peak)
    MOP   = np.vstack((matrix, LOP))
    return MOP

def FFT(y_array_TD,samp_rate,Peak_thresh):
    
    y_array_FD= np.fft.fftshift(np.fft.fft(y_array_TD))
    x_array_FD = np.fft.fftshift(np.fft.fftfreq(len(y_array_FD), 1/samp_rate))
    y_array_FD = y_array_FD/(len(x_array_FD))*2
    MOP = Peak_detection(x_array_FD,y_array_FD,Peak_thresh)

    return x_array_FD, y_array_FD, MOP

def Overlap(func1,func2,dx):
    Ovlap=np.sum(abs(func1)*abs(func2)*dx)
    return Ovlap

def Overlap_new(func1,func2):
    Ovlap=np.sum(np.conj(func1)*func2)
    return Ovlap
    
    






        
    