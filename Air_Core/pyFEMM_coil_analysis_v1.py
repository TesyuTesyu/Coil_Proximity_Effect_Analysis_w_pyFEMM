import femm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import *
import csv

#空芯コイルの抵抗計算.
#pyFEMMで静磁場を計算し、それを用いて理論式（以下の論文中の(6)-(11)）を計算する。
#K. Umetani, S. Kawahara, J. Acero, H. Sarnago, Ó. Lucía and E. Hiraki, "Analytical Formulation of Copper Loss of Litz Wire With Multiple Levels of Twisting Using Measurable Parameters," in IEEE Transactions on Industry Applications, vol. 57, no. 3, pp. 2407-2420, May-June 2021, doi: 10.1109/TIA.2021.3063993.

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

#Dowell法.
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

#-----------------------config-------------------------

sigma=5.8e7#S/m , 銅の導電率.
mu=4*np.pi*1e-7
rho=1/sigma
I_coil=1

alpha_s=1e-3#wire radius.

scale=1e-3#変更できない.

r_in=10#内径[mm]
r_out=13#外径[mm]
z_bottom=0#下の座標[mm], 基本的にゼロ.
z_top=20#上の座標[mm], z_bottom=0 なら高さ.

#静磁界を読むときの分割数. 巻き線が円であることを表現できる程度に、巻き数より十分大きいほうがいい.
num_x=100#横方向.
num_y=300#縦方向.

turn_layer=20#１層当たりの巻き数.
num_layer=2#層数.

#結果データを入れるファイルパス.
file_path_csv_write=r"C:\FEMM_origin\pyFEMM_coil_v1.csv"

#------------------------------------------------------
x_air=r_in/2
y_air=(z_top+z_bottom)/2

x_coil=(r_out+r_in)/2
y_coil=(z_top+z_bottom)/2

#pyFEMMでポスト処理(ほぼデベロッパーのチュートリアルと同じ)
#https://www.femm.info/wiki/pyfemm
# The package must be initialized with the openfemm command.
femm.openfemm()

# We need to create a new Magnetostatics document to work on.
femm.newdocument(0)

# Define the problem type.  Magnetostatic; Units of mm; Axisymmetric; 
# Precision of 10^(-8) for the linear solver; a placeholder of 0 for 
# the depth dimension, and an angle constraint of 30 degrees
freq=0
femm.mi_probdef(freq, 'millimeters', 'axi', 1.e-8, 0, 30)

# Draw a rectangle for the coil;
femm.mi_drawrectangle(r_in, z_bottom, r_out, z_top)

# Define an "open" boundary condition using the built-in function:
femm.mi_makeABC()

# Add block labels, one to each the steel, coil, and air regions.
femm.mi_addblocklabel(x_air,y_air)
femm.mi_addblocklabel(x_coil,y_coil)

# Add some block labels materials properties
femm.mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
femm.mi_addmaterial('Coil', 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)

# Add a "circuit property" so that we can calculate the properties of the
# coil as seen from the terminals.
femm.mi_addcircprop('icoil', I_coil, 1)

# Apply the materials to the appropriate block labels

femm.mi_selectlabel(x_coil,y_coil)
femm.mi_setblockprop('Coil', 0, 1, 'icoil', 0, 0, int(turn_layer*num_layer))
femm.mi_clearselected()

femm.mi_selectlabel(x_air,y_air)
femm.mi_setblockprop('Air', 0, 1, '<None>', 0, 0, 0)
femm.mi_clearselected()

# Now, the finished input geometry can be displayed.
femm.mi_zoomnatural()

# We have to give the geometry a name before we can analyze it.
femm.mi_saveas('coil.fem')

# Now,analyze the problem and load the solution when the analysis is finished
femm.mi_analyze()
femm.mi_loadsolution()

matx=np.linspace(r_in,r_out,num_x)
maty=np.linspace(z_bottom,z_top,num_y)

matHr=np.zeros([num_y,num_x])
matHz=np.zeros([num_y,num_x])
x_all=np.zeros([num_y,num_x])
y_all=np.zeros([num_y,num_x])
i=0
for x in matx:
    j=0
    for y in maty:
        hr, hz=femm.mo_geth(x,y)
        matHr[j,i]=hr
        matHz[j,i]=hz
        x_all[j,i]=x
        y_all[j,i]=y
        j+=1
    i+=1

H=np.sqrt(matHr**2+matHz**2)
'''
fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
plt.pcolormesh(x_all, y_all, H, cmap="jet")
ax[0,0].plot([r_in,r_out],[z_bottom,z_bottom],"k-")
ax[0,0].plot([r_in,r_out],[z_top,z_top],"k-")
ax[0,0].plot([r_in,r_in],[z_bottom,z_top],"k-")
ax[0,0].plot([r_out,r_out],[z_bottom,z_top],"k-")
cbar = plt.colorbar()
plt.show()
'''
#煩雑なので、mm->m に変更
r_in=r_in*scale
r_out=r_out*scale
z_bottom=z_bottom*scale
z_top=z_top*scale

#磁場分布から導体が存在する部分だけを抜き出す.
mask=np.zeros([num_y,num_x])
numx_circle_rad=int(alpha_s/((r_out-r_in)/(num_x-1)))
numy_circle_rad=int(alpha_s/((z_top-z_bottom)/(num_y-1)))
mati0=np.linspace(numx_circle_rad,num_x-1-numx_circle_rad,num_layer, dtype=int)
matj0=np.linspace(numy_circle_rad,num_y-1-numy_circle_rad,turn_layer, dtype=int)
num_point=[]#各円を構成する点の数.
for i0 in mati0:
    for j0 in matj0:
        num_point2=0
        for i_norm in range(numx_circle_rad*2-1+4):#i_norm: normalized, (i0-numx_circle_rad, j0-numy_circle_rad)を中心とした座標.
            for j_norm in range(numy_circle_rad*2-1+4):
                x_norm=(i_norm-2-numx_circle_rad+1)*(r_out-r_in)/(num_x-1)
                y_norm=(j_norm-2-numy_circle_rad+1)*(z_top-z_bottom)/(num_y-1)
                if x_norm**2+y_norm**2<(alpha_s)**2:
                    i=i0+i_norm-2-numx_circle_rad+1
                    j=j0+j_norm-2-numy_circle_rad+1
                    if i>num_x-1:
                        i=num_x-1
                    elif i<0:
                        i=0
                    if j>num_y-1:
                        j=num_y-1
                    elif j<0:
                        j=0
                    mask[j,i]=1
                    num_point2+=1
        num_point.append(num_point2)
H=H*mask

fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
plt.pcolormesh(x_all, y_all, H, cmap="gray")
cbar = plt.colorbar()
plt.show()

#各巻き線内の磁場分布の二乗平均を計算する.
matr=np.linspace(r_in+alpha_s,r_out-alpha_s,num_layer)
mat_H_eq=np.zeros([turn_layer,num_layer])
k=0
for i in range(num_layer):
    i0=mati0[i]
    r=matr[i]
    for j in range(turn_layer):
        j0=matj0[j]
        H_now=H[j0-numy_circle_rad:j0+numy_circle_rad+1,i0-numx_circle_rad:i0+numx_circle_rad+1]

        '''
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
        plt.pcolormesh(x_all[j0-numy_circle_rad:j0+numy_circle_rad+1,i0-numx_circle_rad:i0+numx_circle_rad+1], y_all[j0-numy_circle_rad:j0+numy_circle_rad+1,i0-numx_circle_rad:i0+numx_circle_rad+1], H_now, cmap="gray")
        cbar = plt.colorbar()
        plt.show()
        '''
        H_eq=np.sum(np.sqrt(H_now**2))/num_point[k]
        if np.isnan(H_eq):
            print(fr"H=nan i={i}")
            mat_H_eq[j,i]=0
        else:
            mat_H_eq[j,i]=H_eq
        k+=1

#それを書き出し.
with open(file_path_csv_write, 'w', newline='') as fil:
    writer = csv.writer(fil)
    writer.writerow([sigma,mu,I_coil,alpha_s,scale,r_in,r_out,z_bottom,z_top,num_x,num_y,turn_layer,num_layer])
    writer.writerow(["i,j,mat_H_eq"])
    for i in range(num_layer):
        for j in range(turn_layer):
            writer.writerow([i,j,mat_H_eq[j,i]])

#近接効果係数を計算.
mat_prox_coeff=[]
mat_ferreira=[]
R_ratio_dowell=[]
matf=np.logspace(2,7,300)
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
        i0=mati0[i]
        r=matr[i]
        for j in range(turn_layer):
            j0=matj0[j]
            H_eq=mat_H_eq[j,i]
            k+=1
            R_eq_div.append((Rs+Gs_r*H_eq**2)*2*np.pi*r)#巻き線を１ターンのコイルがNs個あると近似し、それぞれの抵抗を格納.
        matRDC.append(rho/(np.pi*alpha_s**2)*turn_layer*2*np.pi*r)
    R_eq=sum(R_eq_div)
    Rdc=sum(matRDC)
    mat_prox_coeff.append(R_eq/Rdc)#proposed.
    mat_ferreira.append(Fx)#外部磁界を加味しない近接効果係数.
    R_ratio_dowell.append(dowell_proximity_calcu(f,2*alpha_s,mu,sigma,0.8,1))#Dowell法.

fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
ax[0,0].plot(matf,mat_prox_coeff,"k-")
ax[0,0].plot(matf,mat_ferreira,"r--")
ax[0,0].plot(matf,R_ratio_dowell,"g-.")
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
plt.show()

# When the analysis is completed, FEMM can be shut down.
femm.closefemm()
