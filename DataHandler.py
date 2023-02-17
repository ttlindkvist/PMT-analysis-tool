## For Loading data
## Create a combined trace file if it doesn't exist
## Should load runs into dict for later retrieval
## Should be able to compute absorption spectra, with list of runs as input and save the sum
import numpy as np
import glob, os

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

            PMT_combined_file_name = date_folder+'\\'+run_folder_base+'_channelA_combined.dat'
            PD_combined_file_name =  date_folder+'\\'+run_folder_base+'_channelD_combined.dat'
            if not (os.path.exists(PMT_combined_file_name) and os.path.exists(PMT_combined_file_name)):
                self.combine_run_files(run_folder, date_folder)
            
            PMTdata = np.loadtxt(PMT_combined_file_name)
            PDdata = np.loadtxt(PD_combined_file_name)
            
            #Extract wavelengths as first column in matrix
            wavelengths = PMTdata[0]
            PMTdata = (PMTdata[1:]).transpose()
            PDdata = (PDdata[1:]).transpose()
            
            self.cached_runs[run_folder] = {'wavelengths': wavelengths, 'PMT': PMTdata, 'PD' : PDdata}
            self.compute_absorption(run_folder)
    
    def sum_runs(self, run_folders, run_weights=None, run_scalings=None):
        #First see if all runs are loaded
        self.load_runs(run_folders)
        
        wls = np.array([])
        absorption = np.array([])
        if run_weights == None:
            run_weights = [1]*len(run_folders)
        if run_scalings == None:
            run_scalings = [1]*len(run_folders)
        weights = []
        #Flatten wl and absorption arrays for the runs
        for run_folder, run_weight, run_scaling in zip(run_folders, run_weights, run_scalings):
            absorption = np.concatenate((absorption, self.absorption_spectra[run_folder]['absorption']*run_scaling))
            
            run_wls = self.absorption_spectra[run_folder]['wavelengths']
            wls = np.concatenate((wls, run_wls))
            weights += [run_weight]*len(run_wls)
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
        all_run_folders_string = ''
        for run_folder in run_folders:
            all_run_folders_string += run_folder
        self.absorption_spectra[all_run_folders_string] = {'wavelengths': wl_set, 'absorption':absorption_avg, 'absorption_std': absorption_std}
    
    def compute_absorption(self, run_folder):
        PMTintegrate_start = 1300
        PMTintegrate_end = PMTintegrate_start + 1000
        
        PDintegrate_start = 1300
        PDintegrate_end = PMTintegrate_start + 250
        
        wls      = self.cached_runs[run_folder]['wavelengths']
        PMTdata  = self.cached_runs[run_folder]['PMT']
        PDdata   = self.cached_runs[run_folder]['PD']
        if len(np.atleast_1d(wls)) > 1:
            PMT_zeros = np.mean(PMTdata[:,500:1000], axis=1)
            PD_zeros  = np.mean(PDdata[:,500:1000] , axis=1)
            
            PMT_zeros = PMT_zeros.reshape((len(PMT_zeros), 1))
            PD_zeros = PD_zeros.reshape((len(PD_zeros), 1))
            
            PMT_yields = np.trapz(PMTdata[:,PMTintegrate_start:PMTintegrate_end] - PMT_zeros, axis=1)
            PD_yields =  np.trapz(PDdata[:,PDintegrate_start:PDintegrate_end] - PD_zeros, axis=1)
            
            self.absorption_spectra[run_folder] = {'wavelengths' : wls, 'absorption' : -PMT_yields / (wls * PD_yields) * self.PD_responsivity(wls)}
        else:
            PMT_zeros = np.mean(PMTdata[500:1000])
            PD_zeros  = np.mean(PDdata[500:1000])
            
            PMT_yields = np.trapz(PMTdata[PMTintegrate_start:PMTintegrate_end] - PMT_zeros)
            PD_yields = np.trapz(PDdata[PDintegrate_start:PDintegrate_end] - PD_zeros)
            
            self.absorption_spectra[run_folder] = {'wavelengths' : wls, 'absorption' : -PMT_yields / (wls * PD_yields) * self.PD_responsivity(wls)}
        
    def combine_run_files(self, folder, savefolder):
        nfiles = len(glob.glob(folder+'\\channelA*.dat'))
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
            
            PMTtraces.append(np.loadtxt(PMTfile, skiprows=total_header_length))
            PDtraces.append(np.loadtxt(PDfile, skiprows=total_header_length))
            
        PMTtraces = np.array(PMTtraces)
        PDtraces = np.array(PDtraces)
        
        sorted_idxs = np.argsort(wavelengths)
        wavelengths = np.array(wavelengths)[sorted_idxs]
        PMTtraces = PMTtraces[sorted_idxs]
        PDtraces = PDtraces[sorted_idxs]
        
        PMTtraces_and_wl = np.vstack((wavelengths, PMTtraces.transpose()))
        PDtraces_and_wl = np.vstack((wavelengths, PDtraces.transpose()))
        
        run_folder = os.path.basename(folder)
        np.savetxt(savefolder+'\\'+run_folder+'_channelA_combined.dat', PMTtraces_and_wl, header=header_no_wl)
        np.savetxt(savefolder+'\\'+run_folder+'_channelD_combined.dat', PDtraces_and_wl, header=header_no_wl)