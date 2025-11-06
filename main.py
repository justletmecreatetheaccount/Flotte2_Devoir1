import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve1d

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
                
dissipation_rate = kin_viscosity * (np.sum(avg_sq_speed_der['x']) + np.sum(avg_sq_speed_der['y'] + np.sum(avg_sq_speed_der['z'])))

print('dissipation rate : ', dissipation_rate)


dissipation_rate = 0
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

        dissipation_rate += (df['du/dx']**2 / h_sq).mean()

dissipation_rate = 15 * kin_viscosity * (dissipation_rate / 16)

print('worse dissipation rate : ', dissipation_rate)
