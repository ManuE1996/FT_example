import numpy as np

def SD_test(x,y):
    func=np.linalg.det(([x,y,],[y,x]))
    return func

def Mirror_Wave(x,DVR,NOP,Mirror,t):
    while t > 4*DVR:
        t -= 4 * DVR
        
    func = Komplex_Wave(x,t)
    result_func = np.zeros(len(func))
    
    if Mirror == True:
        func = Komplex_Wave(x,0)
        if t <= 2*DVR:
            space = int((2*DVR-t)/(2*DVR/NOP))+1
            result_func[:space] += np.flip(func[:space])
            add = func[space:]
            for k,item in enumerate(add,start=1):
                result_func[k] += item        
            result_func = np.flip(result_func)
        else:
            t -= 2*DVR
            space = int((2*DVR-t)/(2*DVR/NOP))+1
            result_func[:space] += np.flip(func[:space])
            add = func[space:]
            for k,item in enumerate(add,start=1):
                result_func[k] += item        
    else:
        result_func = func
    return result_func

def Single_WP(x,x0,sigma,k0):
    Psi = np.exp(-1/4*(x-x0)**2/sigma**2)*np.exp(1j*k0*x)
    return Psi

def sin_wave(a,f,x,t):
    p = 0
#    t = 0
    x  = a * np.sin(2*np.pi*f*(x-t)+(np.pi/180*p))
    return x

def gauß_wave(x,x0,sigma):
    func = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-x0)**2/(2*sigma**2))
    return func

def Komplex_Wave(x,t):
    # Brauchen die Sinus Terme unbedingt eine innere Zeitentwicklung? Kann x0 = 0 sein?
    mode  = 'Wave_pack'
    x0    = 0
    sigma = 1
    LK    = sin_wave(1,3,x,t) + sin_wave(2,5,x,t)
    SD1   = sin_wave(1,3,x,t) * sin_wave(2,5,x,t)
    SD2   = sin_wave(1,1,x,t) * sin_wave(3,4,x,t)
    func1 = SD1
    func2 = SD2
    if mode == 'Plane_Wave':
        Psi = (func1 + func2)
    if mode == 'Wave_pack':
        Psi = (func1+func2)*np.exp(-1/4*(x-x0)**2/sigma**2)
    if mode == 'gauß_wave':
        Psi = gauß_wave(x,t,sigma)
    if mode == 'SD':
        Psi = np.zeros(len(x))
        for i in range(len(x)):
            Psi[i] = SD_test(sin_wave(1,3,i,t),sin_wave(2,5,i,t))
        
    return Psi
    
def Overlap(func1,func2):
    Ovlap=np.sum(np.conj(func1)*func2)
    return Ovlap

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
    
def Solve_Det(Inp_Mat):
    size = len(Inp_Mat[0,0])
    Determinant = np.zeros(size)
    for i in range(size):
        Determinant[i]=np.linalg.det(Inp_Mat[:,:,i])
    return Determinant

