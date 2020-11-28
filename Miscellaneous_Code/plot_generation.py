import matplotlib.pyplot as plt
import numpy as np

def regularisation_small_model():
    '''Apollo -d2 etc. models'''
    # l2_lambda = [10**(-i) for i in np.linspace(5, 7, 6)]
    # # l2_mses = [4.61e-4, 3.35e-4, 2.53e-4, 2.32e-4, 2.38e-4, 2.61e-4]
    # # l2_maes = [1.08e-2, 9.94e-3, 9.79e-3, 1.01e-2, 1.08e-2, 1.15e-2]
    # l2_mses = [4.61e-4, 3.35e-4, 2.53e-4, 2.32e-4, 2.38e-4, 2.61e-4]
    # l2_maes = [1.08e-2, 9.94e-3, 9.79e-3, 1.01e-2, 1.08e-2, 1.15e-2]
    # dropout = list(np.linspace(0, 0.5, 6))
    # dropout_mses = []
    # dropout_maes = []


    '''Orpheus -d1 etc. models'''
    l2_lambda = [10**(-i) for i in np.linspace(5, 8, 16)]
    l2_mses = [7.78e-4, 6.96e-4, 6.94e-4, 7.17e-4, 6.95e-4, 7.00e-4, 7.26e-4,
                7.39e-4, 7.59e-4, 7.91e-4, 8.22e-4, 8.46e-4, 9.20e-4, 9.79e-4,
                10.03e-4, 17.17e-4]
    l2_maes = [1.18e-2, 1.79e-2, 1.85e-2, 1.92e-2, 1.91e-2, 1.95e-2, 2.01e-2,
                2.02e-2, 2.06e-2, 2.11e-2, 2.15e-2, 2.18e-2, 2.26e-2, 2.33e-2,
                2.36e-2, 3.09e-2]
    dropout = list(np.linspace(0, 0.5, 11))
    dropout_mses = [1.62e-3, 1.08e-3, 0.955e-3, 0.915e-3, 0.866e-3, 0.931e-3,
                    0.817e-3, 0.865e-3, 0.788e-3, 0.816e-3, 0.815e-3]
    dropout_maes = [3.03e-2, 2.43e-2, 2.28e-2, 2.25e-2, 2.18e-2, 2.26e-2, 2.12e-2, 2.18e-2, 2.10e-2, 2.14e-2, 2.14e-2]

    fig = plt.figure()
    ax = plt.subplot(111)
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel('Mean Squared Error', labelpad=20)

    bax1=fig.add_subplot(212, label="1", xlim=[0, 1e-5])
    bax2=fig.add_subplot(212, label="2", frame_on=False, sharey=bax1, xlim=[0, 0.5])

    l2_line, = bax1.plot(l2_lambda, l2_mses, color="C0", label='Model performance with L2 Regularisation')
    bax1.set_xlabel("L2 Regularisation Hyperparameter", color="C0")
    bax1.tick_params(axis='x', colors="C0")

    bax2.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )
    bax1.spines['top'].set_visible(False)
    bax2.spines['top'].set_visible(False)
    plt.ylim([6.5e-4, 11e-4])

    dummy_l2_line, = bax2.plot([100, 400], [100, 400], color="C0", label='L2 Regularisation')
    dropout_line, = bax2.plot(dropout, dropout_mses, color="C1", label='Dropout')


    tax1=fig.add_subplot(211, label="1", xlim=[0, 1e-5])
    tax2=fig.add_subplot(211, label="2", frame_on=False, sharey=tax1, xlim=[0, 0.5])

    l2_line, = tax1.plot(l2_lambda, l2_mses, color="C0", label='Model performance with L2 Regularisation')
    dummy_l2_line, = tax2.plot([100, 400], [100, 400], color="C0", label='L2 Regularisation')
    dropout_line, = tax2.plot(dropout, dropout_mses, color="C1", label='Dropout')
    tax2.plot([0, 1], [1.62e-3, 1.62e-3], color='black', linestyle='dashed', label='No regularisation')

    tax1.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )
    tax1.spines['bottom'].set_visible(False)
    tax2.spines['bottom'].set_visible(False)

    tax2.xaxis.tick_top()
    tax2.set_xlabel('Dropout Hyperparameter', color="C1")
    tax2.xaxis.set_label_position('top')
    tax2.tick_params(axis='x', colors="C1")
    plt.ylim([11.5e-4, 20e-4])


    fig.subplots_adjust(hspace=0.05)

    plt.legend()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    tax1.plot([0, 1], [0, 0], transform=tax1.transAxes, **kwargs)
    bax2.plot([0, 1], [1, 1], transform=bax2.transAxes, **kwargs)

    plt.savefig('/import/tintagel3/snert/david/Honours_Report/Regularisation_plot_small_2.png')
    plt.show()

def ___():
    # If we were to simply plot pts, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax1) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    # plot the same data on both axes
    ax1.plot(pts)
    ax2.plot(pts)

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(.78, 1.)  # outliers only
    ax2.set_ylim(0, .22)  # most of the data

    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.show()

def regularisation_large_model():
    l2_lambda = [10**(-i) for i in np.linspace(6, 8, 11)][1:] + [0]
    l2_mses = [1.75e-3, 1.62e-3, 1.58e-3, 1.70e-3, 1.65e-3, 1.81e-3, 2.23e-3,
                1.92e-3, 1.67e-3, 1.57e-3, 1.47e-3]
    dropout = list(np.linspace(0, 0.5, 11))
    dropout_mses = [1.47e-3, 1.19e-3, 1.25e-3, 1.06e-3, 1.07e-3, 1.05e-3,
                1.07e-3, 1.10e-3, 1.34e-3, 2.42e-3, 2.67e-3]

    fig=plt.figure()
    # plt.suptitle('Mean Squared Errors with varying regularisation hyperparameters')
    ax=fig.add_subplot(111, label="1", xlim=[0, 6e-7])
    ax2=fig.add_subplot(111, label="2", frame_on=False, sharey=ax, xlim=[0, 0.5])

    l2_line, = ax.plot(l2_lambda, l2_mses, color="C0", label='Model performance with L2 Regularisation')
    ax.set_xlabel("L2 Regularisation Hyperparameter", color="C0")
    ax.set_ylabel("Mean Squared Error")
    ax.tick_params(axis='x', colors="C0")
    plt.ylim([1e-3, 3e-3])

    dummy_l2_line, = ax2.plot([100, 400], [100, 400], color="C0", label='L2 Regularisation')
    dropout_line, = ax2.plot(dropout, dropout_mses, color="C1", label='Dropout')
    ax2.plot([0, 1], [1.47e-3, 1.47e-3], color='black', linestyle='dashed', label='No regularisation')
    ax2.xaxis.tick_top()
    ax2.set_xlabel('Dropout Hyperparameter', color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', colors="C1")


    plt.legend()
    plt.tight_layout()

    plt.show()

regularisation_small_model()
# regularisation_large_model()
