import numpy as np
import SpecHeaderReader

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=0)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=0)
    return average, np.sqrt(variance)

class SpecDataHandler:
    #Number of non-intensified pixels on each side
    DEAD_PIXELS = 167
    def __init__(self):
        self.dispersed_fluorescence = {}
    
    def load_runs(self, runs, force_rebinning=True):
        for run in runs:
            # Check if run is already loaded
            if run in self.dispersed_fluorescence.keys():
                continue
            
            header_length, header_dict, data_length = SpecHeaderReader.read_header_string(run)
            
            data = np.loadtxt(run, skiprows=header_length, max_rows=data_length).transpose()
            bin_size = int(1024/data_length)

            if data_length == 1024:
                bin_size = 4
                rebinned_wls = data[0].reshape(-1, bin_size).mean(axis=1)
                rebinned_signal = data[3].reshape(-1, bin_size).sum(axis=1)
                self.dispersed_fluorescence[run] = {'wavelengths'  : rebinned_wls[int(self.DEAD_PIXELS/bin_size):-int(self.DEAD_PIXELS/bin_size)], 
                                                    'fluorescence' : rebinned_signal[int(self.DEAD_PIXELS/bin_size):-int(self.DEAD_PIXELS/bin_size)],
                                                    'molecule' : header_dict.get('Ion name', '')}
            else:
                self.dispersed_fluorescence[run] = {'wavelengths'  : data[0,int(self.DEAD_PIXELS/bin_size):-int(self.DEAD_PIXELS/bin_size)], 
                                                    'fluorescence' : data[3,int(self.DEAD_PIXELS/bin_size):-int(self.DEAD_PIXELS/bin_size)],
                                                    'molecule' : header_dict.get('Ion name', '')}
            
    def sum_runs(self, runs, data_key, run_weights=None, run_scalings=None):
        #First see if all runs are loaded
        self.load_runs(runs)
        
        wls = np.array([])
        signal = np.array([])
        if run_weights is None:
            run_weights = [1]*len(runs)
        if run_scalings is None:
            run_scalings = [1]*len(runs)
        weights = []
        molecules = []

        #Flatten wl and absorption arrays for the runs
        for run_folder, run_weight, run_scaling in zip(runs, run_weights, run_scalings):
            signal = np.concatenate((signal, np.atleast_1d(self.dispersed_fluorescence[run_folder]['fluorescence']*run_scaling)))
            run_wls = np.atleast_1d(self.dispersed_fluorescence[run_folder]['wavelengths'])
            wls = np.concatenate((wls, run_wls))
            
            weights += [run_weight]*len(run_wls)
            molecules.append(self.dispersed_fluorescence[run_folder]['molecule'])
        
        weights = np.array(weights)

        # Sort all arrays wrt. the wl
        sorted_idxs = np.argsort(wls)
        signal_sorted = signal[sorted_idxs]
        weights_sorted = weights[sorted_idxs]
        wl_sorted = wls[sorted_idxs]

        #For multiple measurements at a wl, find the mean and std
        wl_set = np.unique(wl_sorted)
        signal_avg = np.zeros(wl_set.shape)
        signal_std = np.zeros(wl_set.shape)
        for i, wl in enumerate(wl_set):
            vals = signal_sorted[np.argwhere(wl_sorted==wl)]
            ws = weights_sorted[np.argwhere(wl_sorted==wl)]
            avg, std = weighted_avg_and_std(vals, ws)

            signal_avg[i] = avg
            signal_std[i] = std / np.sqrt(len(vals))

        molecules_set = set(molecules)
        molecules_str = ', '.join(molecules_set)
        self.dispersed_fluorescence[data_key] = {'wavelengths': wl_set, 'fluorescence':signal_avg, 
                                                 'fluorescence_std': signal_std, 'molecule':molecules_str}
    

    def auto_rescale_runs(self, run_folders):
        #First see if all runs are loaded
        self.load_runs(run_folders)

        # Determine the overlaps between all pairs
        overlapping_sets = []
        all_wl_ranges = dict()

        for run in run_folders:
            wls = self.dispersed_fluorescence[run]['wavelengths']
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
                reference_wls    = self.dispersed_fluorescence[sorted_runs[i]]['wavelengths']
                reference_signal = self.dispersed_fluorescence[sorted_runs[i]]['fluorescence']
                for run in (x for x in runs_left if x != sorted_runs[i]):
                    wls = self.dispersed_fluorescence[run]['wavelengths']
                    run_wl_start = np.min(wls)
                    run_wl_end = np.max(wls)

                    # Find common wavelengths
                    common_wls_start = max(run_wl_start, np.min(reference_wls))
                    common_wls_end = min(run_wl_end, np.max(reference_wls))

                    print(common_wls_start, common_wls_end)
                    if common_wls_start < common_wls_end:
                        run_signal = self.dispersed_fluorescence[run]['fluorescence']
                        run_start_idx = np.argmin(np.abs(wls - common_wls_start))
                        run_end_idx = np.argmin(np.abs(wls - common_wls_end))

                        run_integral = np.trapz(run_signal[run_start_idx:run_end_idx], wls[run_start_idx:run_end_idx])

                        reference_start_idx = np.argmin(np.abs(reference_wls-common_wls_start))
                        reference_end_idx = np.argmin(np.abs(reference_wls-common_wls_end))

                        reference_integral = np.trapz(reference_signal[reference_start_idx:reference_end_idx], reference_wls[reference_start_idx:reference_end_idx])


                        # additional scaling from reference
                        auto_scale_factor = run_scalings.get(sorted_runs[i],1) * reference_integral / run_integral
                        run_scalings[run] = auto_scale_factor
                        self.dispersed_fluorescence[run]['autoscaling factor'] = auto_scale_factor
                        print(run, auto_scale_factor)
                        rescaled_runs.append(run)


                # Remove rescaled runs from runs_left list
                runs_left = [x for x in runs_left if x not in rescaled_runs]

                i += 1
                if i > maxiter:
                    break
