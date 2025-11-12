import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve1d
from scipy.signal import savgol_filter
import scipy.integrate as integrate
import scipy.fft as fft
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
                    # power spectral density : np.abs(np.fft.fft(df['u']))**2 https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density 
                    # Wiener–Khinchin theorem : the rest
                    # np.real n'est normalement pas necessaire mais https://stackoverflow.com/questions/47850760/using-scipy-fft-to-calculate-autocorrelation-of-a-signal-gives-different-answer
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['u']))**2))
                    f_r = autocorr / autocorr[0]
    
                    # indices where f_r is negative | Suggestion IA, fait sens https://www.researchgate.net/publication/253210572_Autocorrelation_Functions_and_the_Determination_of_Integral_Length_with_Reference_to_Experimental_and_Numerical_Data
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

total_iterations = 4 * 4

with tqdm(total=total_iterations, desc="Structure functions") as pbar:
    for x in range(4) :
        for y in range(4) :
            path = 'Data/pencils_x/x_' + str(x) +'_' + str(y) + '.txt'
            df = pd.read_csv(path, sep=' ', decimal='.', names=['u','v','w'])

            R11 = np.real(np.fft.ifft(np.abs(np.fft.fft(df['u']))**2)) / 32768
            R22 = 0.5 * np.real(np.fft.ifft(np.abs(np.fft.fft(df['v']))**2)) / 32768
            R22 += 0.5 * np.real(np.fft.ifft(np.abs(np.fft.fft(df['w']))**2)) / 32768
            

            d11_for_pencils[x][y] = (2 * (R11[0] - R11))[:maxdr]
            d22_for_pencils[x][y] = (2 * (R22[0] - R22))[:maxdr]

            # u = df['u'].to_numpy()
            # v = df['v'].to_numpy()
            # w = df['w'].to_numpy()
        
            # for r in range(1, maxdr) :
            #     uxr_ux = np.roll(u, -r) - u
            #     vxr_vx = np.roll(v, -r) - v
            #     wxr_wx = np.roll(w, -r) - w
            #     d11_for_pencils[x][y][r] = (uxr_ux**2).mean()
            #     d22_for_pencils[x][y][r] = 0.5 * (vxr_vx**2).mean() + 0.5 * (wxr_wx**2).mean()

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


# --- Code for the Compensated Structure Function Plot ---

# IMPORTANT: You must get these values from your class notes!
# I am using common placeholder values here.
C2_from_notes = 2.1  # <-- !! REPLACE THIS with the value for C2 from your notes
C2_prime_from_notes = (4/3) * C2_from_notes # Or use the C'2 value from your notes

# 1. Calculate the compensation factor
# We use (epsilon_bar * r) raised to the power of (-2/3)
compensation_factor = (dissipation_rate * r)**(-2/3)

# 2. Calculate the compensated structure functions
comp_d11 = d11 * compensation_factor
comp_d22 = d22 * compensation_factor

# 3. Plot the results in log-log scale
plt.figure(figsize=(10, 7))
plt.loglog(r_norm, comp_d11, 'b-', label=r'$D_{11} (\bar{\epsilon} r)^{-2/3}$ (Longitudinal)')
plt.loglog(r_norm, comp_d22, 'r-', label=r'$D_{22} (\bar{\epsilon} r)^{-2/3}$ (Transverse)')

# 4. Add horizontal lines for the theoretical constants
plt.axhline(y=C2_from_notes, color='blue', linestyle='--', 
            label=f'Theoretical $C_2 = {C2_from_notes}$')
plt.axhline(y=C2_prime_from_notes, color='red', linestyle='--', 
            label=f"Theoretical $C'_2 = {C2_prime_from_notes:.2f}$")

# Formatting
plt.xlabel(r'$r/\eta$', fontsize=14)
plt.ylabel(r'Compensated Functions', fontsize=14)
plt.title(r'Compensated Structure Functions', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)

# Limit the x-axis to the inertial range (r/eta >> 1) to see the plateau
# You may need to adjust these limits to best show your plateau
plt.xlim(left=10, right=5e3) 
plt.ylim(bottom=0.5, top=5) # Adjust this y-range to center on your constants

plt.tight_layout()
plt.show()

#
#
# One-dimensional energy spectra
#
#
# pg 196 R11/u'^2 = f(r) R22/u'^2 = g(r) utiliser u11 au lieu de u' psq les u varient bcp
# donc pg 225
# 

total_iterations = 4 * 4
k = fft.fftfreq(32768, d = 2 * np.pi / 32768) * 2 * np.pi

f_r = np.zeros(32768)
g_r = np.zeros(32768)

with tqdm(total=total_iterations, desc="Energy") as pbar:
    for x in range(4) :
        for y in range (4) :
            path = 'Data/pencils_x/x_' + str(x) +'_' + str(y) + '.txt'
            df = pd.read_csv(path, sep=' ', decimal='.', names=['u','v','w'])


            # Convolution theorem
            # power spectral density : np.abs(np.fft.fft(df['u']))**2 https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density 
            # Wiener–Khinchin theorem : the rest
            # np.real n'est normalement pas necessaire mais https://stackoverflow.com/questions/47850760/using-scipy-fft-to-calculate-autocorrelation-of-a-signal-gives-different-answer
            autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['u']))**2))
            f_r += autocorr / autocorr[0]
            autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['v']))**2))
            g_r += autocorr / autocorr[0]
            autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['w']))**2))
            g_r += autocorr / autocorr[0]

                
            pbar.update()

f_r = f_r / 16
g_r = g_r / 32


E11 = 2 * np.real(fft.fft(f_r * rms_u**2)) / 32768
E22 = 2 * np.real(fft.fft(g_r * rms_v**2)) / 32768


plot_limit_points = int(500 * kolmo_scale / h) 
r_physical = np.arange(0, plot_limit_points) * h
r_norm = r_physical / kolmo_scale

# --- 2. Plot f(r) and g(r) ---
plt.figure(figsize=(10, 6))
plt.plot(r_norm, f_r[:plot_limit_points], 'b-', label=r'$f(r)$ (Longitudinal)')
plt.plot(r_norm, g_r[:plot_limit_points], 'r-', label=r'$g(r)$ (Transverse)')

plt.xlabel(r'$r/\eta$', fontsize=14)
plt.ylabel('Autocorrelation', fontsize=14)
plt.title('Longitudinal and Transverse Autocorrelation Functions', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlim(0, 500) # As requested by prompt
plt.ylim(bottom=min(np.min(f_r[:plot_limit_points]), np.min(g_r[:plot_limit_points])), top=1.05)
plt.show()


k_norm = k * kolmo_scale

# --- 2. Get Normalization Factor ---
# This is the unusual normalization requested by the prompt
kolmogorov_norm = (dissipation_rate * kin_viscosity**5)**(1/4)
E11_norm = E11 / kolmogorov_norm
E22_norm = E22 / kolmogorov_norm

# --- 3. Define Theoretical Lines ---
# WARNING: The prompt says to compare with C1 and C2 "as defined in the notes."
# The standard values are C1 ~ 0.5 and C2 = (4/3)*C1.
# YOU MUST CHECK YOUR NOTES for the values you are supposed to use.
C1_const = 0.53  # <-- CHECK YOUR NOTES FOR THIS VALUE
C2_const = (4/3) * C1_const

# We'll plot the theory in a reasonable inertial range (e.g., 10^-3 to 10^-1)
k_theory = np.logspace(-4, 0.5, 50)
theory_C1 = C1_const * (k_theory**(-5/3))
theory_C2 = C2_const * (k_theory**(-5/3))

# --- PLOT A: Dimensionless Energy Spectra (log-log) ---
plt.figure(figsize=(10, 7))
# Plot from k_norm[1] to skip the k=0 mean energy
plt.loglog(k_norm[:2**14], E11_norm[:2**14], 'b-', label=r'$E_{11} / (\overline{\epsilon} \nu^5)^{1/4}$')
plt.loglog(k_norm[:2**14], E22_norm[:2**14], 'r-', label=r'$E_{22} / (\overline{\epsilon} \nu^5)^{1/4}$')

# Plot theoretical slopes
plt.loglog(k_theory, theory_C1, 'k--', label=fr'$C_1 (k\eta)^{{-5/3}}$ (C1 $\approx$ {C1_const})')
plt.loglog(k_theory, theory_C2, 'k:', label=fr'$C_2 (k\eta)^{{-5/3}}$ (C2 $\approx$ {C2_const:.2f})')

plt.title('Dimensionless 1D Energy Spectra', fontsize=16)
plt.xlabel(r'$k\eta$', fontsize=14)
plt.ylabel(r'$E_{ii} / (\overline{\epsilon} \nu^5)^{1/4}$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# --- PLOT B: Compensated Spectra ---
# Calculate compensated values
E11_comp = (k_norm[:2**14]**(5/3)) * E11_norm[:2**14]
E22_comp = (k_norm[:2**14]**(5/3)) * E22_norm[:2**14]

plt.figure(figsize=(10, 7))
# Plot on semilog-x to see the plateau
plt.loglog(k_norm[:2**14], E11_comp, 'b-', label=r'$(k\eta)^{5/3} E_{11} / (\overline{\epsilon} \nu^5)^{1/4}$')
plt.loglog(k_norm[:2**14], E22_comp, 'r-', label=r'$(k\eta)^{5/3} E_{22} / (\overline{\epsilon} \nu^5)^{1/4}$')

# Plot horizontal lines for C1 and C2
plt.axhline(C1_const, color='k', linestyle='--', label=fr'$C_1 \approx {C1_const}$')
plt.axhline(C2_const, color='k', linestyle=':', label=fr'$C_2 \approx {C2_const:.2f}$')

plt.title('Compensated 1D Energy Spectra', fontsize=16)
plt.xlabel(r'$k\eta$', fontsize=14)
plt.ylabel('Compensated Spectra', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()


#
#
# One-dimensional energy spectra accounting for noise
#
#
# pg 196 R11/u'^2 = f(r) R22/u'^2 = g(r) utiliser u11 au lieu de u' psq les u varient bcp
# donc pg 225
# 

total_iterations = 4 * 4 * 3
k = fft.fftfreq(32768, d = 2 * np.pi / 32768) * 2 * np.pi

f_r = np.zeros(32768)
g_r = np.zeros(32768)

with tqdm(total=total_iterations, desc="Energy") as pbar:
    for coord in ['x', 'y', 'z'] :
        for x in range(4) :
            for y in range (4) :
                path = 'Data/pencils_'+ coord + '/'+ coord + '_' + str(x) +'_' + str(y) + '.txt'
                df = pd.read_csv(path, sep=' ', decimal='.', names=['u','v','w'])

                if coord == 'x' :
                    # Convolution theorem
                    # power spectral density : np.abs(np.fft.fft(df['u']))**2 https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density 
                    # Wiener–Khinchin theorem : the rest
                    # np.real n'est normalement pas necessaire mais https://stackoverflow.com/questions/47850760/using-scipy-fft-to-calculate-autocorrelation-of-a-signal-gives-different-answer
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['u']))**2))
                    f_r += autocorr / autocorr[0]
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['v']))**2))
                    g_r += autocorr / autocorr[0]
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['w']))**2))
                    g_r += autocorr / autocorr[0]

                elif coord == 'y' :
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['v']))**2))
                    f_r += autocorr / autocorr[0]
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['u']))**2))
                    g_r += autocorr / autocorr[0]
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['w']))**2))
                    g_r += autocorr / autocorr[0]

                else :
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['w']))**2))
                    f_r += autocorr / autocorr[0]
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['v']))**2))
                    g_r += autocorr / autocorr[0]
                    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(df['u']))**2))
                    g_r += autocorr / autocorr[0]

                
                pbar.update()

f_r = f_r / 48
g_r = g_r / 96

E11 = 2 * np.real(fft.fft(f_r * rms_u**2)) / 32768
E22 = 2 * np.real(fft.fft(g_r * rms_v**2)) / 32768


k_norm = k * kolmo_scale

# --- 2. Get Normalization Factor ---
# This is the unusual normalization requested by the prompt
kolmogorov_norm = (dissipation_rate * kin_viscosity**5)**(1/4)
E11_norm = E11 / kolmogorov_norm
E22_norm = E22 / kolmogorov_norm

# --- 3. Define Theoretical Lines ---
# WARNING: The prompt says to compare with C1 and C2 "as defined in the notes."
# The standard values are C1 ~ 0.5 and C2 = (4/3)*C1.
# YOU MUST CHECK YOUR NOTES for the values you are supposed to use.
C1_const = 0.53  # <-- CHECK YOUR NOTES FOR THIS VALUE
C2_const = (4/3) * C1_const

# We'll plot the theory in a reasonable inertial range (e.g., 10^-3 to 10^-1)
k_theory = np.logspace(-4, 0.5, 50)
theory_C1 = C1_const * (k_theory**(-5/3))
theory_C2 = C2_const * (k_theory**(-5/3))

# --- PLOT A: Dimensionless Energy Spectra (log-log) ---
plt.figure(figsize=(10, 7))
# Plot from k_norm[1] to skip the k=0 mean energy
plt.loglog(k_norm[:2**14], E11_norm[:2**14], 'b-', label=r'$E_{11} / (\overline{\epsilon} \nu^5)^{1/4}$')
plt.loglog(k_norm[:2**14], E22_norm[:2**14], 'r-', label=r'$E_{22} / (\overline{\epsilon} \nu^5)^{1/4}$')

# Plot theoretical slopes
plt.loglog(k_theory, theory_C1, 'k--', label=fr'$C_1 (k\eta)^{{-5/3}}$ (C1 $\approx$ {C1_const})')
plt.loglog(k_theory, theory_C2, 'k:', label=fr'$C_2 (k\eta)^{{-5/3}}$ (C2 $\approx$ {C2_const:.2f})')

plt.title('Dimensionless 1D Energy Spectra', fontsize=16)
plt.xlabel(r'$k\eta$', fontsize=14)
plt.ylabel(r'$E_{ii} / (\overline{\epsilon} \nu^5)^{1/4}$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# --- PLOT B: Compensated Spectra ---
# Calculate compensated values
E11_comp = (k_norm[:2**14]**(5/3)) * E11_norm[:2**14]
E22_comp = (k_norm[:2**14]**(5/3)) * E22_norm[:2**14]

plt.figure(figsize=(10, 7))
# Plot on semilog-x to see the plateau
plt.loglog(k_norm[:2**14], E11_comp, 'b-', label=r'$(k\eta)^{5/3} E_{11} / (\overline{\epsilon} \nu^5)^{1/4}$')
plt.loglog(k_norm[:2**14], E22_comp, 'r-', label=r'$(k\eta)^{5/3} E_{22} / (\overline{\epsilon} \nu^5)^{1/4}$')

# Plot horizontal lines for C1 and C2
plt.axhline(C1_const, color='k', linestyle='--', label=fr'$C_1 \approx {C1_const}$')
plt.axhline(C2_const, color='k', linestyle=':', label=fr'$C_2 \approx {C2_const:.2f}$')

plt.title('Compensated 1D Energy Spectra', fontsize=16)
plt.xlabel(r'$k\eta$', fontsize=14)
plt.ylabel('Compensated Spectra', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

