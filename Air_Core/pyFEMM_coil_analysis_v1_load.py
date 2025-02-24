import matplotlib.pyplot as plt
import numpy as np
from scipy.special import *
import csv

#pyFEMM_coil_analysis_v1.py が書き出したファイルを読み出す.

def f_beiber(x):
    bei_x=bei(x)
    ber_x=ber(x)
    beip_x=beip(x)#Derivative of the Kelvin function bei.
    berp_x=berp(x)
    bei2_x=jv(2, x * np.exp(3 * np.pi * 1j / 4)).imag
    ber2_x=jv(2, x * np.exp(3 * np.pi * 1j / 4)).real
    Fx=x/2*(ber_x*beip_x-berp_x*bei_x)/(berp_x**2+beip_x**2)
    Kx=-x*(ber2_x*berp_x+bei2_x*beip_x)/(ber_x**2+bei_x**2)
    return Fx,Kx

def dowell_proximity_calcu(f,d,myu,sigma,eta_coil,m):
    delta=1/np.sqrt(np.pi*f*myu*sigma)
    zeta=np.sqrt(np.pi)/2*d/delta
    zeta_prime=zeta*np.sqrt(eta_coil)
    if zeta_prime<20:
        SINH=np.sinh(2*zeta_prime)
        COSH=np.cosh(2*zeta_prime)
        SIN=np.sin(2*zeta_prime)
        COS=np.cos(2*zeta_prime)
        R_ratio=zeta_prime*((SINH+SIN)/(COSH-COS)+eta_coil**2*2/3*(m**2-1)*(SINH-SIN)/(COSH+COS))
    else:
        R_ratio=zeta_prime*(1+eta_coil**2*2/3*(m**2-1))
    return R_ratio

#結果データが入っているフォルダパス.
folder_path=r"C:\FEMM_origin"

#ファイル名. カンマで区切っていくつも入れると、それぞれのデータが同時にplotされる.
mat_file_name=["pyFEMM_coil_v1.csv",
"pyFEMM_coil_v1_2.csv",
"pyFEMM_coil_v1_3.csv"]

#plotの色.ファイル名の数だけ定義.
matcol=["k-","r--","g-."]

#周波数.logspaceなので、引数は10^x. logspace(min,max,num)
matf=np.logspace(3,7,300)


fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
jhjhk=0
for file_name in mat_file_name:
    file_path=folder_path+"\\"+file_name
    skiprow=0
    with open(file_path, 'r', encoding='utf-8') as file:
        csvreader = csv.reader(file, delimiter=',')
        i=0
        for row in csvreader:
            if i==0:
                values = [float(x) for x in row]
                [sigma,mu,I_coil,alpha_s,scale,r_in,r_out,z_bottom,z_top,num_x,num_y,turn_layer,num_layer]=values
                turn_layer=int(turn_layer)
                num_layer=int(num_layer)
                mat_H_eq=np.zeros([turn_layer,num_layer])
            elif i==1:
                print(row)
                break
            i+=1
        for row in csvreader:
            values = [float(x) for x in row]
            i=int(values[0])
            j=int(values[1])
            mat_H_eq[j,i]=values[2]

    num_x=int(num_x)
    num_y=int(num_y)

    rho=1/sigma
    #eta_coil=4*alpha_s**2*turn_layer*num_layer/((r_out-r_in)*(z_top-z_bottom))
    eta_coil=0.5

    matr=np.linspace(r_in+alpha_s,r_out-alpha_s,num_layer)
    mat_prox_coeff=[]
    mat_ferreira=[]
    R_ratio_dowell=[]
    for f in matf:
        delta=1/np.sqrt(np.pi*f*mu*sigma)
        gamma_s=2*alpha_s/delta/np.sqrt(2)
        Fx,Kx=f_beiber(gamma_s)
        Rs=rho/(np.pi*alpha_s**2)*Fx
        Gs_r=4*np.pi*rho*Kx#線に直角な磁場による損失成分.
        R_eq_div=[]
        matRDC=[]
        k=0
        for i in range(num_layer):
            r=matr[i]
            for j in range(turn_layer):
                H_eq=mat_H_eq[j,i]
                k+=1
                R_eq_div.append((Rs+Gs_r*H_eq**2)*2*np.pi*r)#巻き線を１ターンのコイルがNs個あると近似し、それぞれの抵抗を格納.
            matRDC.append(rho/(np.pi*alpha_s**2)*turn_layer*2*np.pi*r)
        R_eq=sum(R_eq_div)
        Rdc=sum(matRDC)
        mat_prox_coeff.append(R_eq/Rdc)
        mat_ferreira.append(Fx)
        R_ratio_dowell.append(dowell_proximity_calcu(f,2*alpha_s,mu,sigma,eta_coil,num_layer))

    ax[0,0].plot(matf,mat_prox_coeff,matcol[jhjhk])
    #ax[0,0].plot(matf,mat_ferreira,"r--")
    #ax[0,0].plot(matf,R_ratio_dowell,"g-.")
    jhjhk+=1
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
plt.show()