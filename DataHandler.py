## For Loading data
## Create a combined trace file if it doesn't exist
## Should load runs into dict for later retrieval
## Should be able to compute absorption spectra, with list of runs as input and save the sum
import numpy as np
import glob, os
from PMTHeaderReader import read_header_string

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

class DataHandler:
    
    def __init__(self):
        self.cached_runs = {}
        self.absorption_spectra = {}
    
        poly_fit = np.polyfit(pd_wl, respons, 3)
        self.PD_responsivity = np.poly1d(poly_fit)
        
    def load_runs(self, run_folders):
        #iterate over all folders
        # if a combined run file doesn't exist - create one
        #Load the combined run files into the dict self.loaded_runs
        for run_folder in run_folders:
            # Check if run is already loaded
            if run_folder in self.cached_runs.keys():
                continue
            # Check if the combined traces for the run exists
            run_folder_base = os.path.basename(run_folder)
            date_folder = os.path.dirname(run_folder)

            PMT_combined_file_name = date_folder+'\\'+run_folder_base+'_channelA.dat'
            PD_combined_file_name =  date_folder+'\\'+run_folder_base+'_channelD.dat'
            if not (os.path.exists(PMT_combined_file_name) and os.path.exists(PMT_combined_file_name)):
                self.combine_run_files(run_folder, date_folder)
            
            # Find header length
            # Load info from header
            header_length, header_dict = read_header_string(PMT_combined_file_name)

            PMTdata = np.loadtxt(PMT_combined_file_name, skiprows=header_length)
            PDdata = np.loadtxt(PD_combined_file_name, skiprows=header_length)
            
            #Extract wavelengths as first column in matrix
            wavelengths = PMTdata[0]
            PMTdata = (PMTdata[1:]).transpose()
            PDdata = (PDdata[1:]).transpose()
            
            self.cached_runs[run_folder] = {'wavelengths': wavelengths, 'PMT': PMTdata, 'PD' : PDdata,
                                            'molecule' : header_dict.get('Molecule', '')}
            self.compute_absorption(run_folder)
    
    def sum_runs(self, run_folders, data_key, run_weights=None, run_scalings=None):
        #First see if all runs are loaded
        self.load_runs(run_folders)
        
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
    
    def compute_absorption(self, run_folder):
        PMTintegrate_start = 1300
        PMTintegrate_end = PMTintegrate_start + 1000
        
        PDintegrate_start = 1300
        PDintegrate_end = PMTintegrate_start + 250
        
        wls      = self.cached_runs[run_folder]['wavelengths']
        PMTdata  = self.cached_runs[run_folder]['PMT']
        PDdata   = self.cached_runs[run_folder]['PD']
        molecule   = self.cached_runs[run_folder]['molecule']
        if len(np.atleast_1d(wls)) > 1:
            PMT_zeros = np.mean(PMTdata[:,500:1000], axis=1)
            PD_zeros  = np.mean(PDdata[:,500:1000] , axis=1)
            
            PMT_zeros = PMT_zeros.reshape((len(PMT_zeros), 1))
            PD_zeros = PD_zeros.reshape((len(PD_zeros), 1))
            
            PMT_yields = np.trapz(PMTdata[:,PMTintegrate_start:PMTintegrate_end] - PMT_zeros, axis=1)
            PD_yields =  np.trapz(PDdata[:,PDintegrate_start:PDintegrate_end] - PD_zeros, axis=1)
            
            self.absorption_spectra[run_folder] = {'wavelengths' : wls, 'absorption' : -PMT_yields / (wls * PD_yields) * self.PD_responsivity(wls),
                                                   'molecule': molecule}
        else:
            PMT_zeros = np.mean(PMTdata[500:1000])
            PD_zeros  = np.mean(PDdata[500:1000])
            
            PMT_yields = np.trapz(PMTdata[PMTintegrate_start:PMTintegrate_end] - PMT_zeros)
            PD_yields = np.trapz(PDdata[PDintegrate_start:PDintegrate_end] - PD_zeros)
            
            self.absorption_spectra[run_folder] = {'wavelengths' : wls, 'absorption' : -PMT_yields / (wls * PD_yields) * self.PD_responsivity(wls),
                                                   'molecule': molecule}
        
    def combine_run_files(self, folder, savefolder):
        nfiles = len(glob.glob(folder+'\\channelA[0-9][0-9][0-9]_sum.dat'))
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
            PDtraces.append( np.loadtxt(PDfile, skiprows=total_header_length).flatten())
            
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

    def auto_rescale_runs(self, run_folders):
        #First see if all runs are loaded
        self.load_runs(run_folders)

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
            # Sort runs in order from largest to smallest range
            sorted_runs = sorted(s, key=lambda run: all_wl_ranges[run][1]-all_wl_ranges[run][0])[::-1]

            run_scalings = dict()

            runs_left = sorted_runs[1:]
            
            i = 0
            maxiter = len(sorted_runs)-1
            while len(runs_left) > 0:
                # Rescale all runs with respect to the i'th largest ranged run
                rescaled_runs = []
                reference_wls = self.absorption_spectra[sorted_runs[i]]['wavelengths']
                reference_abs = self.absorption_spectra[sorted_runs[i]]['absorption']
                for run in (x for x in runs_left if x != sorted_runs[i]):
                    wls = self.absorption_spectra[run]['wavelengths']
                    run_wl_start = np.min(wls)
                    run_wl_end = np.max(wls)

                    # Find common wavelengths
                    common_wls_start = max(run_wl_start, np.min(reference_wls))
                    common_wls_end = min(run_wl_end, np.max(reference_wls))

                    print(common_wls_start, common_wls_end)
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
                        print(run, auto_scale_factor)
                        rescaled_runs.append(run)


                # Remove rescaled runs from runs_left list
                runs_left = [x for x in runs_left if x not in rescaled_runs]

                i += 1
                if i > maxiter:
                    break