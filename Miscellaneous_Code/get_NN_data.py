import numpy as np
# r.saveForNN(datapath + 'NN_data.npz')
scan_name = 'BigOne1'
datapath = '/import/silo4/snert/david/Neural_Nets/Data/{}/'.format(scan_name)
output_coeffs = [i[1:] for i in self.allCoeffs]
np.savez(datapath + 'NN_data.npz', zernikes=output_coeffs, fluxes=self.allFinalFluxVals)
