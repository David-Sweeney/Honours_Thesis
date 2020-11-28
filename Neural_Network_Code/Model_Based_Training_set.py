import tensorflow.keras as K
import numpy as np
from Zernike import get_filepath


model_name = 'Apollo_model-b1'
filepath = '{}/Neural_Nets/Models/{}/{}.h5'.format(get_filepath(), model_name.split('_')[0], model_name)
model = K.models.load_model(filepath)

zernikes = (np.random.rand(int(1e3), 10)*2-1)/5
fluxes = model.predict(zernikes)

np.savez('{}/Neural_Nets/Data/Generated11Z/NN_data.npz'.format(get_filepath()), zernikes=zernikes, fluxes=fluxes)
