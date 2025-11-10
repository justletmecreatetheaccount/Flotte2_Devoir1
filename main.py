import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve1d
import scipy.integrate as integrate
import matplotlib.pyplot as plt

#constants
kin_viscosity = 1.10555e-5
L = 2 * np.pi
N_grid = 2**15
h = L / N_grid
h_sq = h**2

# compute the turbulent kinetic energy
avg_sq_speed = np.zeros(3)
rms_u = 0
rms_v = 0
rms_w = 0

total_iterations = 4 * 4 * 3

#progress bar
with tqdm(total=total_iterations, desc="Turbulent kinetic energy") as pbar:
    for coord in ['x','y','z'] :
        for x in range(4) :
            for y in range (4) :
                path = 'Data/pencils_' + coord + '/' + coord + '_' + str(x) +'_' + str(y) + '.txt'
                df = pd.read_csv(path, sep=' ', decimal='.', names=['u','v','w'])
                df['u_sq'] = df['u']**2
                df['v_sq'] = df['v']**2
                df['w_sq'] = df['w']**2
                u_sq_av = df['u_sq'].mean()
                v_sq_av = df['v_sq'].mean()
                w_sq_av = df['w_sq'].mean()

                # compute the turbulent kinetic energy per pencil
                avg_sq_speed[0] += u_sq_av
                avg_sq_speed[1] += v_sq_av
                avg_sq_speed[2] += w_sq_av
                #RMS values
                rms_u += u_sq_av
                rms_v += v_sq_av
                rms_w += w_sq_av

                pbar.update()
            
avg_sq_speed = avg_sq_speed / 48
avg_tu_kin_e = (1/2) * (avg_sq_speed[0] + avg_sq_speed[1] + avg_sq_speed[2])

rms_speed = ((2/3)*avg_tu_kin_e)**(1/2)

rms_u = (rms_u/48)**(1/2)
rms_v = (rms_v/48)**(1/2)
rms_w = (rms_w/48)**(1/2)

print('k : ', avg_tu_kin_e)
print('u_rms : ', rms_u)
print('v_rms : ', rms_v)
print('w_rms : ', rms_w)


avg_sq_speed_der = {'x': np.zeros((3)), 'y': np.zeros((3)), 'z': np.zeros((3))}


#progress bar
with tqdm(total=total_iterations, desc="Dissipation rate") as pbar:
    for coord in ['x','y','z'] :
        for x in range(4) :
            for y in range (4) :
                path = 'Data/pencils_' + coord + '/' + coord + '_' + str(x) +'_' + str(y) + '.txt'
                df = pd.read_csv(path, sep=' ', decimal='.', names=['u','v','w'])
                
                kernel = [-1/12, 2/3, 0, -2/3, 1/12]
                df['du/d' + coord] = convolve1d(
                    df['u'],
                    weights=kernel,
                    mode='wrap'  # periodic boundary
                )
                df['dv/d' + coord] = convolve1d(
                    df['v'],
                    weights=kernel,
                    mode='wrap'  # periodic boundary
                )
                df['dw/d' + coord] = convolve1d(
                    df['w'],
                    weights=kernel,
                    mode='wrap'  # periodic boundary
                )

                df['du/d' + coord + '_sq'] = df['du/d' + coord]**2 / h_sq
                df['dv/d' + coord + '_sq'] = df['dv/d' + coord]**2 / h_sq
                df['dw/d' + coord + '_sq'] = df['dw/d' + coord]**2 / h_sq

                avg_sq_speed_der[coord][0] += df['du/d' + coord + '_sq'].mean()
                avg_sq_speed_der[coord][1] += df['dv/d' + coord + '_sq'].mean()
                avg_sq_speed_der[coord][2] += df['dw/d' + coord + '_sq'].mean()


                pbar.update()


avg_sq_speed_der['x'] = avg_sq_speed_der['x'] / 16
avg_sq_speed_der['y'] = avg_sq_speed_der['y'] / 16
avg_sq_speed_der['z'] = avg_sq_speed_der['z'] / 16
                
dissipation_rate = kin_viscosity * (np.sum(avg_sq_speed_der['x']) + np.sum(avg_sq_speed_der['y']) + np.sum(avg_sq_speed_der['z']))

print('dissipation rate : ', dissipation_rate)

mean_longitudinal_grad_sq = (avg_sq_speed_der['x'][0] +  # du/dx from x-pencils
                             avg_sq_speed_der['y'][1] +  # dv/dy from y-pencils
                             avg_sq_speed_der['z'][2]) / 3 # dw/dz from z-pencils

epsilon_compliant = 15 * kin_viscosity * mean_longitudinal_grad_sq
print('Strictly compliant epsilon:', epsilon_compliant)

worse_dissipation_rate = 0
for x in tqdm(range(4), desc='worse Dissipation rate') :
    for y in range(4) :
        path = 'Data/pencils_x/x_' + str(x) +'_' + str(y) + '.txt'
        df = pd.read_csv(path, sep=' ', decimal='.', names=['u','v','w'])

        kernel = [-1/12, 2/3, 0, -2/3, 1/12]
        df['du/dx'] = convolve1d(
            df['u'],
            weights=kernel,
            mode='wrap'  # periodic boundary
        )

        worse_dissipation_rate += (df['du/dx']**2 / h_sq).mean()

worse_dissipation_rate = 15 * kin_viscosity * (worse_dissipation_rate / 16)

print('worse dissipation rate : ', worse_dissipation_rate)

#Integral length scale
integral_lengths = {'x': np.zeros((4,4)), 'y': np.zeros((4,4)), 'z': np.zeros((4,4))}

#Plot the autocorrelation functions

with tqdm(total=total_iterations, desc="Integral length") as pbar:
    for coord in ['x','y','z'] :
        for x in range(4) :
            for y in range (4) :
                path = 'Data/pencils_' + coord + '/' + coord + '_' + str(x) +'_' + str(y) + '.txt'
                df = pd.read_csv(path, sep=' ', decimal='.', names=['u','v','w'])

                if coord == 'x' :
                    # Convolution theorem
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['u']))**2))
                    f_r = autocorr / autocorr[0]
    
                    # 2. Find first zero crossing | Suggere par IA
                    # indices where f_r is negative
                    neg_indices = np.where(f_r < 0)[0]
    
                    if neg_indices.size > 0:
                        idx_stop = neg_indices[0]
                    else:
                        # Fallback if it never crosses zero (rare)
                        idx_stop = len(f_r) // 2
        
                    # Integrate ONLY up to the first zero crossing
                    integral_lengths['x'][x][y] = integrate.simpson(f_r[:idx_stop], dx=h)                    
                    
                elif coord == 'y' :
                    # Convolution theorem
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['v']))**2))
                    f_r = autocorr / autocorr[0]
    
                    # 2. Find first zero crossing | Suggere par IA
                    # indices where f_r is negative
                    neg_indices = np.where(f_r < 0)[0]
    
                    if neg_indices.size > 0:
                        idx_stop = neg_indices[0]
                    else:
                        # Fallback if it never crosses zero (rare)
                        idx_stop = len(f_r) // 2
        
                    # Integrate ONLY up to the first zero crossing
                    integral_lengths['y'][x][y] = integrate.simpson(f_r[:idx_stop], dx=h)                    
                    
                else :
                    # Convolution theorem
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['w']))**2))
                    f_r = autocorr / autocorr[0]
    
                    # 2. Find first zero crossing | Suggere par IA
                    # indices where f_r is negative
                    neg_indices = np.where(f_r < 0)[0]
    
                    if neg_indices.size > 0:
                        idx_stop = neg_indices[0]
                    else:
                        # Fallback if it never crosses zero (rare)
                        idx_stop = len(f_r) // 2
        
                    # Integrate ONLY up to the first zero crossing
                    integral_lengths['z'][x][y] = integrate.simpson(f_r[:idx_stop], dx=h)                    

                pbar.update()


# Source: Pope, S. B. (2000). Turbulent Flows. Cambridge University Press.

#     While Pope defines the theoretical scale with an integral to ∞, he notes in experimental/computational sections that practical noise and finite domains require truncating the integral, typically at the first zero-crossing, to avoid integrating noise tail oscillations.

# Source: Tritton, D. J. (1988). Physical Fluid Dynamics.

#     Discusses that "negative regions" in the correlation function often arise from conservation of mass in confined or periodic flows, and including them would distort the measurement of the "characteristic eddy size."
# 
L_ux = integral_lengths['x'].mean()
L_vy = integral_lengths['y'].mean()
L_wz = integral_lengths['z'].mean()

print('ux integral length scale : ', L_ux)
print('vy integral length scale : ', L_vy)
print('wz integral length scale : ', L_wz)

integral_length = (L_ux + L_vy + L_wz) / 3
print('integral length scale : ', integral_length)

print('Reynold number : ', (L_ux + L_vy + L_wz)*rms_speed / (3*kin_viscosity))

kolmo_scale = ((kin_viscosity**3)/dissipation_rate)**(1/4)
print('Kolmogorov scale : ', kolmo_scale)

taylor_scale = ((15*kin_viscosity*rms_speed**2)/dissipation_rate)**(1/2)
print('Taylor micro scale : ', taylor_scale)

print('Taylor scale Reynold number : ', taylor_scale*rms_speed / kin_viscosity)


#
#
# Structure functions
#
#
maxdr = math.ceil(5*(10**3)*0.0001770333/h) + 1
print('Max dr : ', maxdr, ' x ', h)

d11_for_pencils = np.zeros((4, 4, maxdr))
d22_for_pencils = np.zeros((4, 4, maxdr))

total_iterations = 4 * 4 * (maxdr - 1)

with tqdm(total=total_iterations, desc="Integral length") as pbar:
    for x in range(4) :
        for y in range(4) :
            path = 'Data/pencils_x/x_' + str(x) +'_' + str(y) + '.txt'
            df = pd.read_csv(path, sep=' ', decimal='.', names=['u','v','w'])
            u = df['u'].to_numpy()
            v = df['v'].to_numpy()
            w = df['w'].to_numpy()
        
            for r in range(1, maxdr) :
                uxr_ux = np.roll(u, -r) - u
                vxr_vx = np.roll(v, -r) - v
                wxr_wx = np.roll(w, -r) - w
                d11_for_pencils[x][y][r] = (uxr_ux**2).mean()
                d22_for_pencils[x][y][r] = 0.5 * (vxr_vx**2).mean() + 0.5 * (wxr_wx**2).mean()

                pbar.update()

d11 = np.mean(np.mean(d11_for_pencils, axis=0), axis=0)
d22 = np.mean(np.mean(d22_for_pencils, axis=0), axis=0)


r = np.arange(1, len(d11) + 1) * h

# 2. Normalize by Kolmogorov scale
r_norm = r / kolmo_scale

# 3. Plot in log-log scale
plt.figure(figsize=(8, 6))
plt.loglog(r_norm, d11, 'b-', label=r'$D_{11}$ (Longitudinal)')
plt.loglog(r_norm, d22, 'r-', label=r'$D_{22}$ (Transverse)')

# Formatting
plt.xlabel(r'$r/\eta$', fontsize=14)
plt.ylabel(r'Structure Functions', fontsize=14)
plt.title(r'Longitudinal and Transverse Structure Functions', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)

# Optional: Limit x-axis to the requested range (e.g., up to 5x10^3)
plt.xlim(left=r_norm[0], right=5e3)

plt.tight_layout()
plt.show()
