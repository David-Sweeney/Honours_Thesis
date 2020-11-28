import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import socket
import subprocess
from shutil import copyfile
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_filepath():
    computer_name = socket.gethostname()
    if computer_name == 'tauron':
        return '/media/tintagel/david'
    elif computer_name == 'caprica':
        return 'Z:'
    else:
        return '/import/tintagel3/snert/david'

def load_FLD(file_path):
    X = pd.read_csv(file_path, skiprows=4, header=None, delim_whitespace=True)
    Xarr = np.asarray(X)
    FLDampl = Xarr[:, ::2]
    FLDintens = Xarr[:, ::2]**2
    FLDphase = Xarr[:, 1::2]
    return FLDampl, FLDphase, FLDintens, FLDphase

def load_MON(file_path, plot=False, save_plot=False):
    # X = pd.read_csv(self.datapath+filename+'.mon', skiprows=5,header=None, delim_whitespace=True)
    X = pd.read_csv(file_path, skiprows=5,header=None, delim_whitespace=True)
    MONdata = np.asarray(X.loc[:, 1:])
    MONposn = np.asarray(X.loc[:, 0])
    if plot:
        plt.figure()
        plt.plot(MONposn, MONdata, linewidth=1.0)
        plt.xlabel('Z position (microns)')
        plt.ylabel('Relative flux')
        plt.title('Monitor values')
        if save_plot:
            plt.savefig(file_path[:-3] + 'png')
        else:
            plt.show()
    return MONdata

def final_fluxes(file_path, useNPts=100):
    MONdata = load_MON(file_path)
    f = MONdata[-useNPts:, :]
    finalFluxVals = f.mean(axis=0)
    return finalFluxVals

def flatten_amplitude(fld_real, fld_imag):
    close_field = fld_real + 1J * fld_imag
    far_field = np.fft.fft2(close_field)
    flat_far_field = np.ones(far_field.shape) + 1J * far_field.imag
    flat_close_field = np.fft.ifft2(flat_far_field)
    return flat_close_field.real, flat_close_field.imag

def average_field(current_fld_real, current_fld_imag):
    old_fld_real, old_fld_imag, _, _ = load_FLD('{}/Reverse_Injection/Saxton/Fields/0.fld'.format(get_filepath()))
    new_fld_real = (current_fld_real + old_fld_real)/2
    new_fld_imag = (current_fld_imag + old_fld_imag)/2
    return new_fld_real, new_fld_imag

def initialise_powers(funnel_indicies):
    powers = [0]*19
    for funnel_index in funnel_indicies:
        powers[funnel_index - 1] = 1 #/ len(funnel_indicies)
        break
    return powers

def redistribute_power(file_path, previous_file_path, funnel_indicies):
    # Load in power from previous
    if file_path is not None:
        powers = final_fluxes(file_path)
        print('Powers from last forward propagation:', powers,
            'Current funnel effectiveness {0:.1%}'.format(sum([powers[i-1] for i in funnel_indicies])))
        return powers
    else:
        # If no previous propagation, initialise fresh array
        return initialise_powers(funnel_indicies)

    if previous_file_path is not None:
        previous_powers = final_fluxes(previous_file_path)
    else:
        # If this ins only second backward propagation, compare to initialisation
        previous_powers = initialise_powers(funnel_indicies)

    # Take away half of the power of each non-funnel waveguide
    for i in range(1, len(powers)+1):
        if i in funnel_indicies:
            continue
        # Reduce the power of this waveguide by half of whatever the shift was
        powers[i-1] -= abs(powers[i-1] - np.average([powers[i-1], previous_powers[i-1]]))
        if powers[i-1] < 0:
            powers[i-1] = 0

    # Calculates power which has been removed or lost
    power_to_redistribute = 1 - sum(powers)
    print(f'Redistributing {power_to_redistribute:.1%} of the light')

    # Split this power evenly among funnel waveguides
    for i in funnel_indicies:
        powers[i-1] += power_to_redistribute/len(funnel_indicies)

    return powers

def combine_amplitude_phase(amplitude_path, phase_path):
    fld_amp, _, _, _ = load_FLD(amplitude_path)
    if phase_path is None:
        return fld_amp, np.zeros(fld_amp.shape)
    _, fld_phase, _, _ = load_FLD(amplitude_path)
    return fld_amp, fld_phase

def conjugate_field(fld_imag):
    return fld_imag * -1.0

def write_IND(powers, bat_path=False):
    # powers is a list of powers to provide to each waveguide

    # Launch field dictionary which stores locations as (x, y)
    launch_field_locations = {1:('0', '0'), 2:('A*(-1/2+0)', 's*-1'), 3:('A*(-1/2+1)', 's*-1'),
        4:('A*(0/2+1)', 's*0'), 5:('A*(1/2+0)', 's*1'), 6:('A*(1/2+-1)', 's*1'),
        7:('A*(0/2+-1)', 's*0'), 8:('A*(-1/2+-1)', 's*-1'), 9:('A*(-2/2+0)', 's*-2'),
        10:('0', 's*-2'), 11:('A*(-2/2+2)', 's*-2'), 12:('A*(1/2+1)', 's*-1'),
        13:('A*(0/2+2)', 's*0'), 14:('A*(1/2+1)', 's*1'), 15:('A*(-2/2+2)', 's*2'),
        16:('0', 's*2'), 17:('A*(-2/2+0)', 's*2'), 18:('A*(1/2+-2)', 's*1'),
        19:('A*(0/2+-2)', 's*0')}

    # Read in basic reverse photonic lantern
    with open('{}\\Reverse_Injection\\RSoft_Models\\Reverse_PL_no_launch_fields.ind'.format(get_filepath()), 'r') as file:
        base_PL = file.read()

    # Write out reverse photonic lantern with appropriate launch fields on each waveguide
    with open('{}\\Reverse_Injection\\Saxton\\Reverse\\Reverse_PL_from_launch_fields.ind'.format(get_filepath()), 'w') as file:
        file.write(base_PL)
        file.write('\n\n')
        for i in range(1, 20):
            x, y = launch_field_locations[i]
            power = powers[i-1]
            file.write(f'''launch_field {i}
                launch_pathway = {i}
                launch_type = LAUNCH_WGMODE
            	launch_normalization = 2
            	launch_align_file = 1
            	launch_width = 13.5
            	launch_height = 13.5
            	launch_position = {x}
            	launch_position_y = {y}
            	launch_power = {power}
            end launch_field\n''')

    if bat_path:
        # Creating bat file to allow this ind to be run in RSoft from Python
        with open(bat_path, 'w') as file:
            # Standard formatting, taken from ZernikePSF.py
            file.write(f'bsimw32 Reverse_PL_from_launch_fields.ind prefix=from_launch_fields wait=0')

def write_FLD(fld_real, fld_imag, out_path, bat_path=False):
    size_data = 165
    len_data = fld_real.shape[0]

    whole_inFLD = np.zeros([len_data, 2 * len_data])  # Empty array to take all data 3d

    # RSoft takes data in columns of real and imaginary parts. (make odd ones imag, even real)
    whole_inFLD[:, ::2] = fld_real
    whole_inFLD[:, 1::2] = fld_imag

    header = ('/rn,a,b/nx0\n/rn,qa,qb\n{0} -{1} {1} 0 OUTPUT_REAL_IMAG_3D\n{0} -{1} {1}').format(len_data, size_data)
    np.savetxt(out_path, whole_inFLD, fmt='%.18E', header=header, comments='')

    if bat_path:
        # Creating bat file to allow this fld to be run in RSoft from Python
        ind_file_name = 'Standard_PL' if 'Forward' in bat_path else 'Reverse_PL_from_file'
        with open(bat_path, 'w') as file:
            # Standard formatting, taken from ZernikePSF.py
            file.write(f'bsimw32 {ind_file_name}.ind prefix=conjugated launch_file=conjugated_inputfield.fld wait=0')
    pass

def plot_FLD(fld_real, fld_imag, fld_intens, fld_phase):
    # Near field amplitude
    plt.figure()
    #plt.imshow(fld_phase[600:720,600:720])
    # plt.imshow(fld_intens[1200:1550,1200:1550])
    # plt.imshow(fld_intens[700:950,700:950])
    plt.imshow(fld_intens[575:750,575:750])
    # plt.imshow(fld_intens)
    plt.colorbar()
    # plt.show()

    # Far field amplitude
    ef = fld_real + 1J * fld_imag
    zim = np.fft.fft2(ef)
    zim = np.fft.fftshift(zim)
    plt.figure()
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
    left=False, right=False, labelbottom=False, labelleft=False)
    # plt.imshow(np.abs(zim[700:950,700:950]))
    # plt.imshow(np.abs(zim[1300:1450,1300:1450]))
    amp = plt.imshow(np.abs(zim[575:750,575:750]))
    plt.title('Far Field Amplitude')
    ax_ff_amp = plt.gca()
    cax_ff_amp = make_axes_locatable(ax_ff_amp).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(amp, cax=cax_ff_amp).ax.set_ylabel('Intensity (arbitrary units)')
    plt.savefig('/import/tintagel3/snert/david/far_field_amplitude.png')
    # plt.show()

    # Far field phase
    efs = np.fft.fftshift(ef)
    zims = np.fft.fft2(efs)
    zim_phase = np.fft.fftshift(np.angle(zims))
    plt.figure()
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
    left=False, right=False, labelbottom=False, labelleft=False)
    # plt.imshow(zim_phase[700:950,700:950])
    # plt.imshow(zim_phase[1300:1450,1300:1450])
    phase = plt.imshow(zim_phase[575:750,575:750], cmap='twilight')
    plt.title('Far Field Phase')
    ax_ff_phase = plt.gca()
    cax_ff_phase = make_axes_locatable(ax_ff_phase).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(phase, cax=cax_ff_phase).ax.set_ylabel('Radians')
    plt.savefig('/import/tintagel3/snert/david/far_field_phase.png')
    plt.show()
    pass

def run_RSoft(bat_path, rsoft_output_path, save_path):
    subprocess.run(bat_path, cwd=bat_path.replace('\\run_RSoft.bat', ''))
    # Copy fld file to save path
    copyfile(rsoft_output_path, save_path)
    # Do the same with the mon file
    copyfile(rsoft_output_path[:-3] + 'mon', save_path[:-3] + 'mon')

def create_amplitude_fld(bat_path):
    subprocess.run(bat_path, cwd=bat_path.replace('\\run_RSoft.bat', ''))

def gerchberg_saxton(starting_int, final_int, funnel_indicies):
    base_file_path = '{}\\Reverse_Injection\\Saxton'.format(get_filepath())

    for i in range(starting_int, final_int + 1):
        forward = True if i % 2 == 0 else False
        input_file_path = '{}\\Fields\\{}.{}'
        fld_file_path = input_file_path.format(base_file_path, str(i-1), 'fld')
        print('New iteration beginning, i =', i)

        if forward:
            print('Preparing to propagate on forward PL')

            # Load in the previous output field
            print('Loading in previous electric field')
            fld_real, fld_imag, fld_intens, fld_phase = load_FLD(fld_file_path)

            print('Conjugating Field')
            fld_imag = conjugate_field(fld_imag)

            # print('Flattening Amplitude')
            # fld_real, fld_imag = flatten_amplitude(fld_real, fld_imag)

            out_path = '{}\\Forward\\conjugated_inputfield.fld'.format(base_file_path)
            bat_path = '{}\\Forward\\run_RSoft.bat'.format(base_file_path)
            rsoft_output_path = '{}\\Forward\\conjugated.fld'.format(base_file_path)
            print('Writing out to .fld file')
            write_FLD(fld_real, fld_imag, out_path, bat_path=bat_path)
        else:
            print('Preparing to propagate on Reverse PL')

            mon_file_path = input_file_path.format(base_file_path, str(i-1), 'mon')
            previous_mon_file_path = input_file_path.format(base_file_path, str(i-3), 'mon')
            if i - 3 <= 0:
                previous_mon_file_path = None
            if i - 1 <= 0:
                mon_file_path = None
            print('Redistributing power from previous electric field to waveguides', funnel_indicies)
            powers = redistribute_power(mon_file_path, previous_mon_file_path, funnel_indicies)
            print('New distribution of powers', powers)

            bat_path = '{}\\Reverse\\run_RSoft.bat'.format(base_file_path)
            rsoft_output_path = '{}\\Reverse\\from_launch_fields.fld'.format(base_file_path)
            print('Writing out to .ind file')
            write_IND(powers, bat_path)

            print('Running .ind file')
            create_amplitude_fld(bat_path)

            print('Combining amplitude and phase')
            if i - 1 <= 0:
                fld_file_path = None
            fld_amp, fld_phase = combine_amplitude_phase(rsoft_output_path, fld_file_path)

            print('Conjugating Field')
            fld_phase = conjugate_field(fld_phase)

            out_path = '{}\\Reverse\\conjugated_inputfield.fld'.format(base_file_path)
            rsoft_output_path = '{}\\Reverse\\conjugated.fld'.format(base_file_path)
            print('Writing out to .fld file')
            write_FLD(fld_amp, fld_phase, out_path, bat_path=bat_path)

        save_path_fld = '{}\\Fields\\{}.fld'.format(base_file_path, str(i))
        print('Beginning RSoft propagation')
        run_RSoft(bat_path, rsoft_output_path, save_path_fld)
        print('Rsoft propagation complete.\n')

    print('Gerchberg-Saxton complete!')

def main():
    file_path = '{}/Reverse_Injection/Saxton/Fields/2.fld'.format(get_filepath())
    fld_real, fld_imag, fld_intens, fld_phase = load_FLD(file_path)
    fld_imag = conjugate_field(fld_imag)
    # fld_real, fld_imag = flatten_amplitude(fld_real, fld_imag)
    # plot_FLD(fld_real, fld_imag, fld_intens, fld_phase)
    fld_real, fld_imag = average_field(fld_real, fld_imag)
    # plot_FLD(fld_real, fld_imag, fld_intens, fld_phase)
    out_path = '{}/Reverse_Injection/Saxton/Reverse/conjugated_inputfield.fld'.format(get_filepath())
    write_FLD(fld_real, fld_imag, out_path)

def plot_GS():
    file_path = '/import/tintagel3/snert/david/Reverse_Injection/Saxton/Fields/{}.mon'
    for i in range(1, 11):
        load_MON(file_path.format(i), plot=True, save_plot=True)
# main()
# gerchberg_saxton(1, 10, [3, 11, 12])
# plot_GS()

file_path = '{}/Reverse_Injection/Saxton/Central_Funnel/1.fld'.format(get_filepath())
fld_real, fld_imag, fld_intens, fld_phase = load_FLD(file_path)
plot_FLD(fld_real, fld_imag, fld_intens, fld_phase)
