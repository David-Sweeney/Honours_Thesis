from Zernike import ModelInfo, get_filepath
import numpy as np
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

def run_adversarial(model_name):
    npz_file = np.load('{}/Neural_Nets/Models/{}/{}_mse.npz'.format(get_filepath(), model_name.split('_')[0], model_name))
    train_x = npz_file['train_x']
    train_mse = npz_file['train_mse']

    model = K.Sequential()
    for i in range(3):
        model.add(Dense(500))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(optimizer=K.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, min_delta=1e-6, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=40, verbose=1, min_delta=1e-4, mode='min')
    history = model.fit(train_x, train_mse, validation_split=0.2, epochs=2000, batch_size=32)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    plt.title('Loss')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylim([0, np.percentile(history.history['val_loss'], 99)])
    plt.show()

    model.save('{}/Neural_Nets/Models/{}/{}_adversary.h5'.format(get_filepath(), model_name.split('_')[0], model_name))
    pass

def calculate_mse(model, set, set_y):
    # Calculate predictions over the entire set
    predictions = model.predict(set)

    # Calculate the MSE
    set_mse = np.mean((predictions - set_y)**2, axis=1)

    return set_mse

def generate_data(model_name):
    model_info = ModelInfo(
                    datasets = ('BigOne23', ),
                    # transfer_datasets = ('BigOne25', 'BigOne26'),
                    model_name = 'Hestia5_Adversary',
                    suffix = '',
                    model_layers = [500]*3,
                    epochs = 300)
    # Establish the correct model to load
    # Note that here that number of Zernikes will be one more than is tested here as NNs never see the Piston Zernike
    # if model_info.zernikes == 7:

    # Load in the model
    filepath = '{}/Neural_Nets/Models/{}/{}.h5'.format(get_filepath(), model_name.split('_')[0], model_name)
    model = K.models.load_model(filepath)
    model_info.load_data()

    # Establish MSE on trainging, validation and testing sets
    train_mse = calculate_mse(model, model_info.train_x, model_info.train_y)
    val_mse = calculate_mse(model, model_info.val_x, model_info.val_y)
    test_mse = calculate_mse(model, model_info.test_x, model_info.test_y)

    # Save this new data
    np.savez('{}/Neural_Nets/Models/{}/{}_mse.npz'.format(get_filepath(), model_name.split('_')[0], model_name), model_name=model_name, train_x=model_info.train_x,
            val_x=model_info.val_x, test_x=model_info.test_x, train_mse=train_mse, val_mse=val_mse, test_mse=test_mse)

    print('{}/Neural_Nets/Models/{}/{}_mse.npz'.format(get_filepath(), model_name.split('_')[0], model_name))

def screen_training_examples(model_name):
    model = K.models.load_model('{}/Neural_Nets/Models/{}/{}_adversary.h5'.format(get_filepath(), model_name.split('_')[0], model_name))

    # Initialise parameters
    scan_name = 'BigOne30' # ***
    np.random.seed(int(scan_name[-2:]))
    number_of_scans = 4000
    number_of_zerinke_modes = 8
    max_coeff_value = 0.5
    acceptance_fraction = 0.1

    # Sort coeffs using Decorate-Sort-Undecorate
    coeffs = np.random.rand(int(number_of_scans/acceptance_fraction), number_of_zerinke_modes-1)*2*max_coeff_value - max_coeff_value
    predicted_mse = model.predict(coeffs)
    coeffs = coeffs.tolist()
    predicted_mse = predicted_mse.tolist()
    zipped = list(zip(predicted_mse, coeffs))
    zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
    coeffs = np.array([coeff for mse, coeff in zipped])

    # Reduce to number_of_scans
    coeffs = coeffs[:number_of_scans]

    # Concatenate in zero column and save
    coeffs = np.concatenate((np.zeros((number_of_scans, 1)), coeffs), axis=1)
    coeffs_list = coeffs.tolist()
    out_path = '/import/tintagel3/snert/david/Neural_Nets/Data/{}'.format(scan_name)
    os.mkdir(out_path)
    np.savez('{}/adversarial_coeffs.npz'.format(out_path), scan_name=scan_name, coeffs_list=coeffs_list)

    pass

def compare_predictions(model_name):
    model = K.models.load_model('{}/Neural_Nets/Models/{}/{}.h5'.format(get_filepath(), model_name.split('_')[0], model_name))

    def get_mse_from_data(scan_name):
        filename = '{}/Neural_Nets/Data/{}/NN_data.npz'.format(get_filepath(), scan_name)
        npz_file = np.load(filename)
        zernike_data = npz_file['zernikes']
        flux_data = npz_file['fluxes']

        mses = calculate_mse(model, zernike_data, flux_data)
        return mses

    true_mses = get_mse_from_data('BigOne29')
    adversarial_model = K.models.load_model('{}/Neural_Nets/Models/{}/{}_adversary.h5'.format(get_filepath(), model_name.split('_')[0], model_name))

    filename = '{}/Neural_Nets/Data/{}/NN_data.npz'.format(get_filepath(), 'BigOne29')
    npz_file = np.load(filename)
    zernike_data = npz_file['zernikes']
    predicted_mses = adversarial_model.predict(zernike_data)

    npz_file = np.load('{}/Neural_Nets/Models/{}/{}_mse.npz'.format(get_filepath(), model_name.split('_')[0], model_name))
    val_mse = npz_file['val_mse']

    traditional_mses = get_mse_from_data('BigOne25')
    direct_compare_mses = get_mse_from_data('BigOne26')
    noisier_mses = get_mse_from_data('BigOne28')
    selective_mses = get_mse_from_data('BigOne30')

    print('True:', np.mean(true_mses))
    print('Predicted:', np.mean(predicted_mses))
    print('Validation:', np.mean(val_mse))
    print('1x Data Set:', np.mean(traditional_mses))
    print('Other 4x Dataset:', np.mean(direct_compare_mses))
    print('16x Dataset:', np.mean(noisier_mses))
    print('Selective:', np.mean(selective_mses))

    plt.hist(true_mses, density=True, alpha=0.5)
    plt.hist(predicted_mses, density=True, alpha=0.5)
    plt.hist(val_mse, density=True, alpha=0.5)
    plt.hist(traditional_mses, density=True, alpha=0.5)
    plt.hist(direct_compare_mses, density=True, alpha=0.5)
    plt.hist(noisier_mses, density=True, alpha=0.5)
    plt.hist(selective_mses, density=True, alpha=0.5)
    plt.ylabel('Density')
    plt.xlabel('MSE')
    plt.title('MSE Distribution of Different Data Sets')
    plt.legend(['True', 'Predicted', 'Validation', '1x Data Set', 'Other 4x Data Set', '16x Data Set', 'Selective'])
    plt.show()

if __name__ == '__main__':
    model_name = 'Hestia5_model-d47' #***
    # generate_data(model_name)
    # run_adversarial(model_name)
    # screen_training_examples(model_name)
    compare_predictions(model_name)
