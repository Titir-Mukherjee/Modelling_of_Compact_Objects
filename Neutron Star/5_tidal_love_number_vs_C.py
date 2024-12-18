import pandas as pd
import os
import numpy as np
from joblib import Parallel, delayed
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import math


global G,c,alpha
G = 6.6730831e-8
c = 2.99792458e10


# Constants specific to this code
MeV_fm3_to_pa = 1.6021766e35
c_km = 2.99792458e5  # km/s

# Rest of the constants from constants.py
mN = 1.67e-24
mev_to_ergs = 1.602176565e-6
fm_to_cm = 1.0e-13
ergs_to_mev = 1.0 / mev_to_ergs
cm_to_fm = 1.0 / fm_to_cm
Msun = 1.988435e33
MeV_fm3_to_pa_cgs = 1.6021766e33
km_to_mSun = G / c ** 2
hbarc3 = 197.32700288295746 ** 3
nucleon_mass = 938.04
pi = math.pi


data = np.loadtxt("EOS1_Unif-RMF.txt", dtype=float)


Pf = data[:,1]*MeV_fm3_to_pa_cgs  # Convert Pressure (MeV/fm^3) to pascals (Pa)
rho_cgs = data[:,0]* MeV_fm3_to_pa_cgs / c**2  # Convert Energy Density (MeV/fm^3) to g/cm^3
cPf = CubicSpline(rho_cgs, Pf)
crf = CubicSpline(Pf, rho_cgs)
cnf = CubicSpline(Pf, rho_cgs)
crs = CubicSpline(Pf, rho_cgs)
cra = CubicSpline(Pf, rho_cgs)
cPa = CubicSpline(rho_cgs, Pf)
cPs = CubicSpline(rho_cgs, Pf)

def fp(p,bool):
    #dp = p/1e5
    dp = p * 0.005
    if(bool==0):
        #res=(-crs(p+(2*dp))+8*crs(p+dp)-8*crs(p-dp)+crs(p-(2*dp)))/(12*dp)
        res = (-1 / 60 * crf(p - (3*dp)) + 3 / 20 * crf(p - (2*dp))  - 3 / 4 * crf(p - (1*dp)) + 3 / 4 * crf(p + (1*dp))  - 3 / 20 * crf(p + (1*dp)) + 1 / 60 * crf(p + (1*dp))) / dp
    elif(bool==1):
         #res=(-crf(p+(2*dp))+8*crf(p+dp)-8*crf(p-dp)+crf(p-(2*dp)))/(12*dp)
             res = (-1 / 60 * crf(p - (3*dp)) + 3 / 20 * crf(p - (2*dp))  - 3 / 4 * crf(p - (1*dp)) + 3 / 4 * crf(p + (1*dp))  - 3 / 20 * crf(p + (1*dp)) + 1 / 60 * crf(p + (1*dp))) / dp  
    elif(bool==2):
         #res=(-cra(p+(2*dp))+8*cra(p+dp)-8*cra(p-dp)+cra(p-(2*dp)))/(12*dp)
             res = (-1 / 60 * crf(p - (3*dp)) + 3 / 20 * crf(p - (2*dp))  - 3 / 4 * crf(p - (1*dp)) + 3 / 4 * crf(p + (1*dp))  - 3 / 20 * crf(p + (1*dp)) + 1 / 60 * crf(p + (1*dp))) / dp  
    return res



def f(x,bool):
    r=x[0]
    m=x[1]
    P=x[2]
    phi=x[3]
    H=x[4]
    B=x[5]
    if(bool==0):
        rho=crs(P)
        F=crs(P+(rho*c*c))
    elif(bool==1):
        rho=crf(P)
        F=crf(P+(rho*c*c))
        Ayu=crf((rho*c*c)-P)
    elif(bool==2):
        rho=cra(P)
        F=cra(P+(rho*c*c))
    a=fp(P,bool)
    dr_dr=1
    dm_dr =4.0*np.pi*(r**2) * rho  
   
    #dm_dr = 4.0 * np.pi * (r ** 2) * rho - ((r ** 2) / 4.0) * (alpha * (-rho + (3.0 * P) / (c * c)))    (for f(R,T))
   
    dP_dr = -(((G * m * rho) / (r ** 2)) * (1 + (P / (rho * c * c))) * (1 + ((4 * np.pi * P * (r ** 3)) / (m * c * c)))) / (1 - ((2 * G * m) / (r * c * c)))
       
   
   
    #for mod f(R,T) the below dP_dr expression which is commented out
    #dP_dr = -(((G * m * rho) / (r ** 2)) * (1 + (P / (rho * c * c))) * (1 + ((r ** 3) / (2 * m)) * (0.5 * alpha * (-rho + (3 * P) / (c * c)) + alpha * (rho + (P) / (c * c))) +
               #((4 * np.pi * P * (r ** 3)) / (m * c * c)))) / (1 - ((2 * G * m) / (r * c * c)))
    dphi_dr=(-dP_dr)/((rho*c*c)*(1+(P/(rho*c*c))))
   
    dH_dr=B
    component1 = (1 - 2 * G * m / (r * c**2))**(-1) * B * (
    (G / c**2) * ((2 * m) / r**2 + 4 * pi * r * (rho - P / c**2)) - 2 / r)
    component2 = (1 - 2 * G * m / (r * c**2))**(-1) * H * (
    6 / r**2 - 4 * pi * G / c**2 * (a * c* c* (rho + P / c**2) + 5 * rho + 9 * P / c**2)
    + (4 * G**2 / ((c**4) * (r**4))) * (1 - 2 * G * m / (r * c**2))**(-1) * (m + 4 * pi *  P * r**3 / c**2 )**2)
    dB_dr= component1+component2
   
    return np.array([dr_dr, dm_dr, dP_dr, dphi_dr, dH_dr, dB_dr])

def f2(x,y,bool):
    r=x[0]
    M=x[1]
    P=x[2]
    phi=x[3]
    j=y[0]
    w=y[1]
   
    if(bool==0):
        rho=crs(P)
    elif(bool==1):
        rho=crf(P)
    elif(bool==2):
        rho=cra(P)
    dj_dr=(((8.*np.pi)/3.)*(r**4)*(rho+(P/(c**2)))*w*(np.exp(-phi)))*(np.sqrt(1-(2*G*M)/(r*c*c)))
   
    dw_dr=(G*np.exp(phi)*j)/(c*c*(r**4)*(np.sqrt(1-(2*G*M)/(r*c*c))))
   
    return np.array([dj_dr,dw_dr])
   
def ns_solve(rho_0,bool):
#Initial Conditions
    dr=500 #In cm
    if(bool==0):
        P_0=cPs(rho_0)
    elif(bool==1):
        P_0=cPf(rho_0)
    elif(bool==2):
        P_0=cPa(rho_0)
    X=np.zeros([6,80000])
    X[:,0]=np.array([500,1,P_0,0.001,500*500,1000])

    #Solve using RK4
    for i in range(1,80000):
        k1=f(X[:,i-1],bool)
        k2=f(X[:,i-1]+k1*0.5*dr,bool)
        k3=f(X[:,i-1]+k2*0.5*dr,bool)
        k4=f(X[:,i-1]+k3*dr,bool)
   
        X[:,i]=X[:,i-1]+(dr*(k1+2*k2+2*k3+k4))/6.
       
        if((X[2,i]/P_0)<1e-10):
            break

    #for j in range(i,80000):
        #X=np.delete(X,i,1)
   
    alpha=X[3,i-1]
    k=(0.5*np.log(1-((2*G*X[1,i-1])/(X[0,i-1]*c*c))))/alpha
    X[3,]=X[3,]*k
       
    Y=np.zeros([2,i])
    Y[:,0]=np.array([0.001,10])
   
    for j in range(1,i):
        k_1=f2(X[:,j-1],Y[:,j-1],bool)
        k_2=f2(X[:,j-1],Y[:,j-1]+k_1*0.5*dr,bool)
        #k_2 = f(X[:, j-1] + np.concatenate(([0], k_1[1:] * 0.5 * dr)), bool)
        k_3=f2(X[:,j-1],Y[:,j-1]+k_2*0.5*dr,bool)
        k_4=f2(X[:,j-1],Y[:,j-1]+k_3*dr,bool)
       
        Y[:,j]=Y[:,j-1]+(dr*(k_1+2*k_2+2*k_3+k_4))/6.
   
   
    Y[1,i-1]=Y[1,i-1]+((2*G*Y[0,i-1])/((c**2)*(X[0,i-1]**3)))
    return X[:,i-1],Y[:,i-1]

rho_cgs = np.arange(2.5e14, 1e15, 0.5e13)
rho_cgs = np.append(rho_cgs, np.arange(1e15, 4e15, 0.5e14))
res_f1 = np.zeros([6, len(rho_cgs)])
res_f2 = np.zeros([2, len(rho_cgs)])

# Parallel computation
results = Parallel(n_jobs=-1)(delayed(ns_solve)(rho, 1) for rho in rho_cgs)

for i in range(len(rho_cgs)):
    #res_f1[:,i] = ns_solve(rho_cgs[i],1)
    res_f1[:,i],res_f2[:,i]=ns_solve(rho_cgs[i],1)
    print(i)
   
   
R_f=res_f1[0,]/1e5
M_f=res_f1[1,]/2e33
H_f=res_f1[4,]

B_f=res_f1[5,]

y_f=(R_f*1e5*B_f)/H_f

C_f=(2*G*M_f*2e33)/(R_f*1e5*c*c)




w_f=res_f2[1,]
J_f=res_f2[0,]
I_f=np.divide(J_f,w_f)


def k(y,C):
    k2 = (1.6*(C**5)*((1-2*C)**2)*(2+2*C*(y-1)-y))/(2*C*(6-3*y+3*C*(5*y-8))+4*(C**3)*(13-11*y+C*(3*y-2)+2*C**2*(1+y))+3*((1-2*C)**2)*(2-y+2*C*(y-1))*np.log(1-2*C))
    #k2 = (1.6*(C**5)*((1-2*C)**2)*(2+2*C*(y-1)-y))/(2*C*(6-3*y+3*C*(5*y-8))+4*(C**3)*(13-11*y+C*(3*y-2)+2*C*C*(1+y))+3*((1-2*C)**2)*(2-y+2*C*(y-1))*np.log(abs(1-2*C)))
    return k2
   
# Calculate dimensionless tidal deformability L
def dimensionless_L(k2, R, M):
    return (2*k2*R**5) / (3  * (M**5))
#def dimensionless_L(k2, R):
    #return (2 * k2 * R ** 5) / 3
   
# Calculate dimensionless compactness parameter C without G and c
def dimensionless_C(M, R):
    return  (M / R )

# Calculate dimensionless mass without G and c
def dimensionless_M(M):
    return M

# Calculate dimensionless Love number k2

k2_f = k(y_f, C_f)

# Calculate dimensionless tidal deformability L

L_f = dimensionless_L(k2_f, R_f,M_f)

# Calculate dimensionless compactness parameter C

C_f = dimensionless_C(M_f, R_f)

# Calculate dimensionless mass

M_f = dimensionless_M(M_f)    


data_fps = np.column_stack((M_f, R_f, L_f, k2_f,w_f, J_f, I_f))
np.savetxt("Mass-Radi-Tiddef-Rot.txt", data_fps, delimiter='\t', header="M_f\tR_f\tL_f\tk2_f\tw_f\tJ_f\tI_f")

data_fps = np.column_stack((w_f, J_f, I_f))
np.savetxt("Rot_prop.txt", data_fps, delimiter='\t', header="w_f\tJ_f\tI_f") 

import matplotlib.pyplot as plt

# Create a figure with 2 subplots
plt.figure(figsize=(12, 10))

# First subplot: k2 vs Cf
plt.subplot(2, 1, 1)
plt.plot(C_f, k2_f, label='$k_2$ vs $C_f$')
plt.xlabel('Compactness Parameter $C_f$')
plt.ylabel('Love Number $k_2$')
plt.title('Love Number $k_2$ vs Compactness Parameter $C_f$')
plt.grid(True)
plt.legend()

# Second subplot: k2 vs Mf
plt.subplot(2, 1, 2)
plt.plot(M_f, k2_f, label='$k_2$ vs $M_f$')
plt.xlabel('Mass $M_f$')
plt.ylabel('Love Number $k_2$')
plt.title('Love Number $k_2$ vs Mass $M_f$')
plt.grid(True)
plt.legend()

# Show the figure
plt.tight_layout()
plt.show()



