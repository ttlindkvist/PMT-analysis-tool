## For Loading data
## Create a combined trace file if it doesn't exist
## Should load runs into dict for later retrieval
## Should be able to compute absorption spectra, with list of runs as input and save the sum
import numpy as np
import glob, os
from PMTHeaderReader import read_header_string
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from PyQt6.QtWidgets import QMessageBox

respons = np.array([0.083, 0.092, 0.106, 0.121, 0.135, 0.149, 0.163, 0.177, 0.191, 0.205, 0.219, 0.232, 0.246, 0.26, 0.273, 0.286, 0.299, 0.312, 0.325, 0.337, 0.348, 0.360, 0.371, 0.382, 0.393, 0.403, 0.412, 0.42, 0.427, 0.433, 0.437])
pd_wl = np.arange(400, 710, 10)

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=0)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=0)
    return average, np.sqrt(variance)

def load_k_factor_calib_file(filename):
    n_header_lines = 0
    OD = 0
    with open(filename) as file:
        line = file.readline().lower()
        while not "end of header" in line: 
            if "od:" in line: OD = float(line.split()[-1])
            n_header_lines += 1
            line = file.readline().lower()

    wls, k_factor = np.loadtxt(filename, skiprows=n_header_lines+1).T
    if '240617' in filename:
        idxs = wls > 422
        wls = wls[idxs]
        k_factor = k_factor[idxs]
    return wls, k_factor, OD

def calibrate_laser(wls, laser):
    if laser == 'ULLA':
        delta_wl_savgol = np.array([ 0.14052475,  0.08169978,  0.02941339, -0.01650662, -0.0562325 ,-0.08993645, -0.11779071, -0.13996751, -0.16185717, -0.167965  ,-0.17290128, -0.16757357, -0.16173146, -0.14210102, -0.13141432,-0.11321242, -0.11909304, -0.12691767, -0.14785069, -0.17115577,-0.15838315, -0.15206135, -0.15574024, -0.15652816, -0.16250123,-0.15442587, -0.14342506, -0.15319164, -0.17767354, -0.20334141,-0.22902421, -0.23188445, -0.25333809, -0.25950798, -0.23944402,-0.22497392, -0.24676572, -0.25882918, -0.25559261, -0.2488573 ,-0.25465678, -0.25669341, -0.26308971, -0.24615378, -0.24314724,-0.24711819, -0.24606021, -0.22896473, -0.22268813, -0.23938868,-0.25492528, -0.26205301, -0.25550078, -0.22999748, -0.18027201,-0.10105329,  0.0129298 ])
        calib_set_wls = np.arange(420, 705, 5)
        spline = CubicSpline(calib_set_wls, delta_wl_savgol)
        return wls + spline(wls)
    if laser == 'VICTOR':
        delta_wl_savgol = np.array([ 0.41314955,  0.35048426,  0.30667866,  0.2780959 ,  0.2610991 , 0.25442721,  0.24870566,  0.23302188,  0.23380647,  0.23907008, 0.20746168,  0.18160019,  0.14233637,  0.15261759,  0.16684618, 0.18612841,  0.1607958 ,  0.14161541,  0.12549482,  0.09485463, 0.04744147,  0.06439607,  0.07521274,  0.07179225,  0.07623764, 0.04878863,  0.0253984 ,  0.02744024,  0.0144728 ,  0.01541213, 0.02409175,  0.01790054,  0.01158028,  0.01731228,  0.01589966, 0.00587276,  0.0139424 , -0.00697825,  0.00411212, -0.02705211,-0.01375572, -0.01248567, -0.01646521, -0.03458876, -0.07339769,-0.13943338])
        calib_set_wls = np.arange(420, 650, 5)
        spline = CubicSpline(calib_set_wls, delta_wl_savgol)
        return wls + spline(wls)
    return wls

class DataHandler:
    
    def __init__(self):
        self.cached_runs = {}
        self.absorption_spectra = {}
    
        poly_fit = np.polyfit(pd_wl, respons, 3)
        self.PD_responsivity = np.poly1d(poly_fit)
        
    def load_runs(self, run_folders, use_k_factor, k_factor_filename, force_reload=False):
        #iterate over all folders
        # if a combined run file doesn't exist - create one
        #Load the combined run files into the dict self.loaded_runs
        for run_folder in run_folders:
            run_folder_base = os.path.basename(run_folder)
            date_folder = os.path.dirname(run_folder)
            PMT_combined_file_name = date_folder+'\\'+run_folder_base+'_channelA.dat'
            PD_combined_file_name =  date_folder+'\\'+run_folder_base+'_channelD.dat'
            n_files = len(glob.glob(run_folder+'\\channelA*'))
            
            # Check if run is already loaded
            reload = False
            if run_folder in self.cached_runs.keys():
                if n_files == len(np.atleast_1d(self.cached_runs[run_folder]['wavelengths'])):
                    if force_reload: pass 
                    else: continue
                else:
                    reload = True
            
            # Check if the combined traces for the run exists
            if reload or (not (os.path.exists(PMT_combined_file_name) and os.path.exists(PMT_combined_file_name))):
                success = self.combine_run_files(run_folder, date_folder)
                if not success: continue

            # Find header length
            # Load info from header
            header_length, header_dict = read_header_string(PMT_combined_file_name)
            PMTdata = np.loadtxt(PMT_combined_file_name, skiprows=header_length)
            PDdata = np.loadtxt(PD_combined_file_name, skiprows=header_length)
            #Extract wavelengths as first row in matrix
            wavelengths = PMTdata[0]
            
            # Check number of wavelengths is equal to number of _channelA.dat files in folder
            if len(np.atleast_1d(wavelengths)) != n_files:
                self.combine_run_files(run_folder, date_folder)

                # Reload
                header_length, header_dict = read_header_string(PMT_combined_file_name)
                PMTdata = np.loadtxt(PMT_combined_file_name, skiprows=header_length)
                PDdata = np.loadtxt(PD_combined_file_name, skiprows=header_length)
                wavelengths = PMTdata[0]

            PMTdata = (PMTdata[1:]).transpose()
            PDdata = (PDdata[1:]).transpose()
            
            self.cached_runs[run_folder] = {'wavelengths': wavelengths, 'PMT': PMTdata, 'PD' : PDdata,
                                            'molecule' : header_dict.get('Molecule', ''),
                                            'injections' : int(header_dict.get('Traces per scan step')),
                                            'PD OD' : float(header_dict.get('OD @ PD', '0')),
                                            'Laser': header_dict.get('Active laser', '')}
            self.compute_absorption(run_folder, use_k_factor, k_factor_filename)
    
    def sum_runs(self, run_folders, data_key, use_k_factor, k_factor_filename, run_weights=None, run_scalings=None):
        #First see if all runs are loaded
        self.load_runs(run_folders, use_k_factor, k_factor_filename)
        wls = np.array([])
        absorption = np.array([])
        if run_weights is None:
            run_weights = [1]*len(run_folders)
        if run_scalings is None:
            run_scalings = [1]*len(run_folders)
        weights = []
        molecules = []

        #Flatten wl and absorption arrays for the runs
        for run_folder, run_weight, run_scaling in zip(run_folders, run_weights, run_scalings):
            absorption = np.concatenate((absorption, np.atleast_1d(self.absorption_spectra[run_folder]['absorption']*run_scaling)))
            run_wls = np.atleast_1d(self.absorption_spectra[run_folder]['wavelengths'])
            wls = np.concatenate((wls, run_wls))
            
            weights += [run_weight]*len(run_wls)
            molecules.append(self.absorption_spectra[run_folder]['molecule'])
        weights = np.array(weights)

        # Sort all arrays wrt. the wl
        sorted_idxs = np.argsort(wls)
        absorption_sorted = absorption[sorted_idxs]
        weights_sorted = weights[sorted_idxs]
        wl_sorted = wls[sorted_idxs]

        #For multiple measurements at a wl, find the mean and std
        wl_set = np.unique(wl_sorted)
        absorption_avg = np.zeros(wl_set.shape)
        absorption_std = np.zeros(wl_set.shape)
        for i, wl in enumerate(wl_set):
            vals = absorption_sorted[np.argwhere(wl_sorted==wl)]
            ws = weights_sorted[np.argwhere(wl_sorted==wl)]
            avg, std = weighted_avg_and_std(vals, ws)

            absorption_avg[i] = avg
            absorption_std[i] = std / np.sqrt(len(vals))

        molecules_set = set(molecules)
        molecules_str = ', '.join(molecules_set)
        self.absorption_spectra[data_key] = {'wavelengths': wl_set, 'absorption':absorption_avg, 'absorption_std': absorption_std,
                                            'molecule': molecules_str}
    
    def compute_absorption(self, run_folder, use_k_factor, k_factor_filename):
        wls      = self.cached_runs[run_folder]['wavelengths']
        PMTdata  = self.cached_runs[run_folder]['PMT'].T
        PDdata   = self.cached_runs[run_folder]['PD'].T
        molecule   = self.cached_runs[run_folder]['molecule']
        n_injections = self.cached_runs[run_folder]['injections']
        PD_OD = self.cached_runs[run_folder]['PD OD']
        PD_to_power       = 10**PD_OD/self.PD_responsivity(wls)
        PD_to_power_calib = 10**PD_OD/self.PD_responsivity(wls)
        
        PD_max = np.argmax(np.sum(PDdata, axis=1))
        PDintegrate_start = PD_max - 60
        if PDintegrate_start < 1200 or PDintegrate_start > 1400:
            print('ERROR: unexpected PD signal. Max PD signal at: ', PD_max*0.2, 'ns')
            msg = QMessageBox()
            msg.setWindowTitle('PD signal error')
            msg.setText(f'ERROR: PD signal error. See terminal for PD maximum position')
            msg.exec()
            PDintegrate_start = 1300

        PDintegrate_end = PDintegrate_start + 250
        
        PMTintegrate_start = PDintegrate_start
        PMTintegrate_end = PMTintegrate_start + 500 #(in units of 0.2 ns = 100 ns)

        

        PMT_zeros = np.mean(PMTdata[500:1000], axis=0)
        PD_zeros  = np.mean(PDdata[500:1000], axis=0)
        
        PMT_yields = np.trapz(PMTdata[PMTintegrate_start:PMTintegrate_end] - PMT_zeros, axis=0)
        PD_yields = np.trapz(PDdata[PDintegrate_start:PDintegrate_end] - PD_zeros, axis=0)
        
        if use_k_factor:
            k_wls, k_factors, kOD = load_k_factor_calib_file(k_factor_filename)
            isWithinRange = np.min(k_wls) <= np.min(wls) and np.max(k_wls) >= np.max(wls)
            if not isWithinRange: 
                print('ERROR: this calibration file is only valid in the region', np.min(k_wls), '-', np.max(k_wls))
                msg = QMessageBox()
                msg.setWindowTitle('Calibration error')
                msg.setText(f'ERROR: this calibration file is only valid in the region {np.min(k_wls)} - {np.max(k_wls)}')
                msg.exec()
                
            #else:
            filtered_kfactor = savgol_filter(k_factors, int(10/np.diff(k_wls)[0]), 3)
            idxs = np.clip(np.searchsorted(k_wls, wls, side='left'), 0, len(k_wls)-1)
            PD_to_power_calib = 10**(PD_OD) * 10**kOD * filtered_kfactor[idxs]
        
        self.absorption_spectra[run_folder] = {'wavelengths' : wls, 'absorption' : -PMT_yields / (wls * PD_yields * PD_to_power_calib) * n_injections,
                                               'absorption (not power calibrated)' : -PMT_yields / (wls * PD_yields * PD_to_power) * n_injections,
                                               'molecule': molecule,
                                               'PMT yields' : -PMT_yields, 
                                               'PD yields' : PD_yields,
                                               'PD power' : PD_yields*PD_to_power_calib}


        # if len(np.atleast_1d(wls)) > 1:
        #     PMT_zeros = np.mean(PMTdata[:,500:1000], axis=1)
        #     PD_zeros  = np.mean(PDdata[:,500:1000] , axis=1)
            
        #     PMT_zeros = PMT_zeros.reshape((len(PMT_zeros), 1))
        #     PD_zeros = PD_zeros.reshape((len(PD_zeros), 1))
            
        #     PMT_yields = np.trapz(PMTdata[:,PMTintegrate_start:PMTintegrate_end] - PMT_zeros, axis=1)
        #     PD_yields =  np.trapz(PDdata[:,PDintegrate_start:PDintegrate_end] - PD_zeros, axis=1)
            
        #     self.absorption_spectra[run_folder] = {'wavelengths' : wls, 'absorption' : -PMT_yields / (wls * PD_yields * PD_to_power) * n_injections,
        #                                            'molecule': molecule}
        # else:
        #     PMT_zeros = np.mean(PMTdata[500:1000])
        #     PD_zeros  = np.mean(PDdata[500:1000])
            
        #     PMT_yields = np.trapz(PMTdata[PMTintegrate_start:PMTintegrate_end] - PMT_zeros)
        #     PD_yields = np.trapz(PDdata[PDintegrate_start:PDintegrate_end] - PD_zeros)
            
        #     # self.absorption_spectra[run_folder] = {'wavelengths' : wls, 'absorption' : -PMT_yields / (wls * PD_yields) * self.PD_responsivity(wls)* n_injections * 10**(-PD_OD),
        #     self.absorption_spectra[run_folder] = {'wavelengths' : wls, 'absorption' : -PMT_yields / (wls * PD_yields * PD_to_power) * n_injections,
        #                                            'molecule': molecule}
        
    def combine_run_files(self, folder, savefolder):
        nfilesA = len(glob.glob(folder+'\\channelA[0-9][0-9][0-9]_sum.dat'))
        nfilesD = len(glob.glob(folder+'\\channelD[0-9][0-9][0-9]_sum.dat'))
        nfiles = min(nfilesA, nfilesD)
        PMTfiles = [folder+'\\channelA'+str(x).zfill(3)+'_sum.dat' for x in range(nfiles)]
        PDfiles = [folder+'\\channelD'+str(x).zfill(3)+'_sum.dat' for x in range(nfiles)]
        
        PMTtraces = []
        PDtraces = []
        wavelengths = []
        header_no_wl = ''
        total_header_length = 0

        for PMTfile, PDfile in zip(PMTfiles, PDfiles):

            # Save the header and excitation wavelengths
            with open(PMTfile) as f:
                header_no_wl = ''
                line = ''
                total_header_length = 0
                while not ("End of Header" in line):
                    line = next(f)
                    total_header_length += 1
                    if "Excitation wavelength" in line:
                        wavelengths.append(float(line.split()[2]))
                    else:
                        header_no_wl += line
        
            PMTtraces.append(np.loadtxt(PMTfile, skiprows=total_header_length).flatten())
            PDtraces.append(np.loadtxt(PDfile, skiprows=total_header_length).flatten())
            
        PMTtraces = np.array(PMTtraces)
        PDtraces = np.array(PDtraces)
        
        sorted_idxs = np.argsort(wavelengths)
        wavelengths = np.array(wavelengths)[sorted_idxs]
        PMTtraces = PMTtraces[sorted_idxs]
        PDtraces = PDtraces[sorted_idxs]
        
        PMTtraces_and_wl = np.vstack((wavelengths, PMTtraces.transpose()))
        PDtraces_and_wl = np.vstack((wavelengths, PDtraces.transpose()))
        
        run_folder = os.path.basename(folder)
        
        #Remove last newline from loaded header
        header_no_wl = header_no_wl.rstrip('\n')
        np.savetxt(savefolder+'\\'+run_folder+'_channelA.dat', PMTtraces_and_wl, header=header_no_wl, comments='')
        np.savetxt(savefolder+'\\'+run_folder+'_channelD.dat', PDtraces_and_wl, header=header_no_wl, comments='')
        return True
    
    def auto_rescale_runs(self, run_folders, use_k_factor, k_factor_filename):
        #First see if all runs are loaded
        self.load_runs(run_folders, use_k_factor, k_factor_filename)

        # Determine the overlaps between all pairs
        overlapping_sets = []
        all_wl_ranges = dict()

        for run in run_folders:
            wls = self.absorption_spectra[run]['wavelengths']
            run_wl_start = np.min(wls)
            run_wl_end = np.max(wls)
            all_wl_ranges[run] = [run_wl_start, run_wl_end]

            #Compare with previous regions, to find continuous overlaps
            has_overlap = False
            for i, s in enumerate(overlapping_sets):
                #Check overlap in each set
                for set_run in s:
                    set_run_region = all_wl_ranges[set_run]
                    if max(run_wl_start, set_run_region[0]) <= min(run_wl_end, set_run_region[1]):
                        has_overlap = True
                        overlapping_sets[i].add(run)
                        break

            if not has_overlap:
                s = set()
                s.add(run)
                overlapping_sets.append(s)
        
        # Rescale each set - find individual overlaps in each set
        for s in overlapping_sets:            
            # # Sort runs in order from largest to smallest range
            # sorted_runs = sorted(s, key=lambda run: all_wl_ranges[run][1]-all_wl_ranges[run][0])[::-1]

            # Sort runs in order from largest to smallest integral
            sorted_runs = sorted(s, key=lambda run: np.trapz(self.absorption_spectra[run]['absorption'], self.absorption_spectra[run]['wavelengths']))[::-1]

            run_scalings = dict()

            runs_left = sorted_runs[1:]
            
            i = 0
            maxiter = len(sorted_runs)-1
            while len(runs_left) > 0:
                # Rescale all runs with respect to the i'th largest ranged run
                rescaled_runs = []
                reference_wls = self.absorption_spectra[sorted_runs[i]]['wavelengths']
                reference_abs = self.absorption_spectra[sorted_runs[i]]['absorption']
                self.absorption_spectra[sorted_runs[i]]['autoscaling factor'] = 1
                for run in (x for x in runs_left if x != sorted_runs[i]):
                    wls = self.absorption_spectra[run]['wavelengths']
                    run_wl_start = np.min(wls)
                    run_wl_end = np.max(wls)

                    # Find common wavelengths
                    common_wls_start = max(run_wl_start, np.min(reference_wls))
                    common_wls_end   = min(run_wl_end, np.max(reference_wls))

                    # print(common_wls_start, common_wls_end)
                    if common_wls_start < common_wls_end:
                        run_abs = self.absorption_spectra[run]['absorption']
                        run_start_idx = np.argmin(np.abs(wls - common_wls_start))
                        run_end_idx = np.argmin(np.abs(wls - common_wls_end))

                        run_integral = np.trapz(run_abs[run_start_idx:run_end_idx], wls[run_start_idx:run_end_idx])

                        reference_start_idx = np.argmin(np.abs(reference_wls-common_wls_start))
                        reference_end_idx = np.argmin(np.abs(reference_wls-common_wls_end))

                        reference_integral = np.trapz(reference_abs[reference_start_idx:reference_end_idx], reference_wls[reference_start_idx:reference_end_idx])


                        # additional scaling from reference
                        auto_scale_factor = run_scalings.get(sorted_runs[i],1) * reference_integral / run_integral
                        run_scalings[run] = auto_scale_factor
                        self.absorption_spectra[run]['autoscaling factor'] = auto_scale_factor
                        # print(run, auto_scale_factor)
                        rescaled_runs.append(run)


                # Remove rescaled runs from runs_left list
                runs_left = [x for x in runs_left if x not in rescaled_runs]

                i += 1
                if i > maxiter:
                    break