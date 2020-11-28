import numpy as np
import matplotlib.pyplot as plt
def plot_normal(previous=0):
    fluxes = np.arange(19)
    plt.figure(figsize=(10, 10)).tight_layout()
    max_j = 3
    offset = previous*max_j*3
    for j in range(max_j):
        plt.subplot(2,max_j,j+1)
        for i in range(3*j, 3*j+3):
            plt.scatter(fluxes, test_y_predicted[i+offset], color=colours[i], marker='x')
            plt.scatter(fluxes, test_y[i+offset], color=colours[i], marker='.')
            plt.ylim((0, 0.25))
        plt.legend(['Predicted Flux', 'True Flux'], loc='upper right')
        if j == 0:
            plt.ylabel('Flux Value (arb units, normalised to total 1)')
        elif j == 1:
            plt.title('Model Flux Predictions')
        plt.subplot(2,max_j,max_j+j+1)
        if j == 0:
            plt.ylabel('Flux Difference (arb units)')
        elif j == 1:
            plt.title('Difference From Ground Truth')
            plt.xlabel('Output Mode Number')
        for i in range(3*j, 3*j+3):
            plt.scatter(fluxes, test_y_predicted[i+offset] - test_y[i+offset], color=colours[i], marker='s')
            plt.ylim((-0.04, 0.04))
    plt.suptitle('first_attempt-l10')
    plt.show()

def plot_reversed(previous=0):
    zernikes = np.arange(1, 7)
    plt.figure(figsize=(50, 10)).tight_layout()
    max_j = 5
    offset = previous*max_j*3
    for j in range(max_j):
        plt.subplot(2,max_j,j+1)
        for i in range(3*j, 3*j+3):
            plt.scatter(zernikes, test_y_predicted[i+offset], color=colours[i], marker='.')
            plt.scatter(zernikes, test_y[i+offset], color=colours[i], marker='x')
            plt.ylim((-0.2, 0.2))
        plt.legend(['Predicted Zernike Mode', 'True Zernike Mode'], loc='upper right')
        if j == 0:
            plt.ylabel('Zernike Value (microns)')
        elif j == 2:
            plt.title('Model Zernike Predictions')
        plt.subplot(2,max_j,max_j+j+1)
        if j == 0:
            plt.ylabel('Zernike Mode Difference (arb units)')
        elif j == 2:
            plt.title('Difference From Ground Truth')
            plt.xlabel('Output Mode Number')
        for i in range(3*j, 3*j+3):
            plt.scatter(zernikes, test_y_predicted[i+offset] - test_y[i+offset], color=colours[i], marker='s')
            plt.ylim((-0.25, 0.25))
    plt.suptitle('first_attempt-r-l10')
    plt.show()

filepath = '/import/silo4/snert/david/Neural_Nets/Models/BigOne1/test_results.npz'
results = np.load(filepath)
test_y = results['test_y']
# test_y = results['test_y']/5
test_y_predicted = results['test_y_predicted']
# test_y_predicted = results['test_y_predicted']/5
colours = ('#000000', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF', '#B22222', '#228B22', '#FF00FF', '#FFD700', '#DAA520', '#808080', '#008000', '#ADFF2F', '#FF69B4', '#CD5C5C', '#4B0082', '#F0E68C', '#E6E6FA', '#7CFC00', '#FFFACD', '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#D3D3D3', '#90EE90', '#FFB6C1', '#FFA07A', '#20B2AA', '#87CEFA', '#778899', '#B0C4DE', '#00FF00', '#32CD32', '#FF00FF', '#800000', '#66CDAA', '#0000CD', '#BA55D3', '#9370D8', '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585', '#191970', '#FFE4E1', '#FFE4B5', '#FFDEAD', '#000080', '#808000', '#6B8E23', '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE', '#D87093', '#FFDAB9', '#CD853F', '#FFC0CB', '#00FFFF', '#DDA0DD', '#B0E0E6', '#800080', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072', '#F4A460', '#2E8B57', '#A0522D', '#C0C0C0', '#87CEEB', '#6A5ACD', '#708090', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE', '#F5DEB3', '#FFFF00', '#9ACD32')
# for i in range(len(test_y_predicted)):
plot_reversed(1)


# plt.savefig('/import/silo4/snert/david/Neural_Nets/Models/BigOne1/test_results1.png')
