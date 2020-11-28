"""
Tool to read and plot RSoft results.

Relevant data files:
bptmp.fld - "Transverse Field Profile at Z=40000" - Note, could also be launch field if this was shown last...
bptmp.mon - "Monitor Value (a.u.)"
bptmp_xz.dat - Side view 1
bptmp_yz.dat - Side view 2
bptmp_**.pdo - Coordinates to draw waveguides over the .dat files

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

class Rsoftdata:
    def __init__(self, datapath):
        self.datapath = datapath
        self.FLDampl = None
        self.FLDintens = None
        self.FLDphase = None
        self.MONdata = None
        self.MONposn = None
        self.XZampl = None
        self.YZampl = None
        self.sourceData = None
        self.finalFluxVals = None
        self.filename = None
        self.allFinalFluxVals = []
        self.allResults = []
        self.allCoeffs = []


    class resultsSet:
        # Class to hold all the results for single simulation
        def __init__(self, FLDampl, FLDintens, FLDphase, MONdata, MONposn, XZampl, YZampl,
                     sourceData, finalFluxVals, coeffs):
            self.FLDampl = FLDampl
            self.FLDintens = FLDintens
            self.FLDphase = FLDphase
            self.MONdata = MONdata
            self.MONposn = MONposn
            self.XZampl = XZampl
            self.YZampl = YZampl
            self.sourceData = sourceData
            self.finalFluxVals = finalFluxVals
            self.coeffs = coeffs

    def loadFLD(self, filename='bptmp'):
        X = pd.read_csv(self.datapath + filename + '.fld', skiprows=4, header=None, delim_whitespace=True)
        Xarr = np.asarray(X)
        self.FLDampl = Xarr[:, ::2]
        self.FLDintens = Xarr[:, ::2]**2
        self.FLDphase = Xarr[:, 1::2]

    def loadMON(self, filename='bptmp'):
        X = pd.read_csv(self.datapath+filename+'.mon', skiprows=5,header=None, delim_whitespace=True)
        self.MONdata = np.asarray(X.loc[:, 1:])
        self.MONposn = np.asarray(X.loc[:, 0])

    def loadXZ(self, filename='bptmp'):
        X = pd.read_csv(self.datapath + filename + '_xz.dat', skiprows=4, header=None, delim_whitespace=True)
        Xarr = np.asarray(X)
        self.XZampl = Xarr

    def loadYZ(self, filename='bptmp'):
        X = pd.read_csv(self.datapath + filename + '_yz.dat', skiprows=4, header=None, delim_whitespace=True)
        Xarr = np.asarray(X)
        self.YZampl = Xarr

    def readall(self, filename='bptmp'):
        self.loadFLD(filename)
        self.loadMON(filename)
        self.loadXZ(filename)
        self.loadYZ(filename)
        self.filename = filename

    def plotall(self, clim=[0,1], linewidth=1.0, fignum=1):
        plt.figure(fignum)
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.imshow(self.FLDampl)
        plt.title('Output field amplitude')
        # plt.imshow(self.FLDampl**2)
        # plt.title('Output field intensity')
        plt.subplot(2, 2, 2)
        plt.plot(self.MONposn, self.MONdata, linewidth=linewidth)
        plt.xlabel('Z position (microns)')
        plt.ylabel('Relative flux')
        plt.title('Monitor values')
        plt.subplot(2, 2, 3)
        plt.imshow(self.XZampl, clim=clim)
        plt.title('XZ Amplitude')
        plt.subplot(2, 2, 4)
        plt.imshow(self.YZampl, clim=clim)
        plt.title('YZ Amplitude')
        fig = plt.gcf()
        fig.suptitle(self.filename)
        plt.show()
        # plt.tight_layout()


    def finalFluxes(self, useNPts=100):
        if useNPts is None:
            self.finalFluxVals = self.MONdata[-1, :]
        else:
            f = self.MONdata[-useNPts:, :]
            self.finalFluxVals = f.mean(axis=0)
        return self.finalFluxVals

    def readMulti(self, scanset='', fileprefix='zernikePSFs', readOne=None, keepAllResults=False,
                  showPlots=True):
        # E.g. scanset = 'focusScan01', fileprefix = 'zernikePSFs'
        npzFilename = self.datapath+scanset + '_metadata.npz'
        npzfile = np.load(npzFilename)
        allInputFilenames = npzfile['allOutfilenames']
        allInputCoeffsList = npzfile['coeffsList']
        if keepAllResults:
            allSourceData = npzfile['allData']

        if readOne is not None:
            allInputFilenames = [allInputFilenames[readOne]]
            allInputCoeffsList = [allInputCoeffsList[readOne]]
            if keepAllResults:
                allSourceData = [allSourceData[readOne]]
        missingFiles = 0
        for k in range(len(allInputFilenames)):
            try:
                inputFilename = allInputFilenames[k]
                coeffs = allInputCoeffsList[k]
                self.readall(inputFilename)
                if showPlots:
                    self.plotall()
                plt.pause(0.001)
                if keepAllResults:
                    self.sourceData = allSourceData[k]
                if showPlots:
                    self.plotSourceData()
                    input('')
                plt.pause(0.001)
                print('Simulation number', k)
                # print(inputFilename)
                print(coeffs)
                finalFluxVals = self.finalFluxes()
                print(finalFluxVals, sum(finalFluxVals))
                self.allFinalFluxVals.append(finalFluxVals)
                self.allCoeffs.append(coeffs)

                if keepAllResults:
                    curResults = self.resultsSet(self.FLDampl, self.FLDintens, self.FLDphase, self.MONdata, self.MONposn,
                                                 self.XZampl, self.YZampl, self.sourceData, finalFluxVals, coeffs)
                    self.allResults.append(curResults)
            except FileNotFoundError:
                print('No file', allInputFilenames[k])
                missingFiles += 1
                continue

        if showPlots:
            self.plotFinalFluxes()
        print(missingFiles, 'files missing.')

    def plotFinalFluxes(self, fignum=3, norm=None, xvals=None, thresh=None):
        plt.figure(fignum)

        if norm == 'total':
            normdFluxes = []
            for fluxvec in self.allFinalFluxVals:
                fluxvecNormd = fluxvec / sum(fluxvec)
                normdFluxes.append(fluxvecNormd)
            finalFluxArr = np.asarray(normdFluxes)
        elif norm == 'each':
            finalFluxArr = copy.deepcopy(np.asarray(self.allFinalFluxVals))
            for k in range(finalFluxArr.shape[1]):
                finalFluxArr[:, k] = finalFluxArr[:, k] / np.max(finalFluxArr[:, k])
        else:
            finalFluxArr = np.asarray(self.allFinalFluxVals)
        self.finalFluxArr = finalFluxArr
        if xvals is None:
            plt.plot(finalFluxArr)
            plt.plot((0, len(self.allFinalFluxVals)), (0, 0), 'k:')
        else:
            plt.plot(xvals, finalFluxArr)
            plt.plot((xvals[0], xvals[-1]), (0, 0), 'k:')

        if thresh is not None:
            pass


    def plotSourceData(self, fignum=2):
        psf_ampl = self.sourceData[0]
        psf_phase = self.sourceData[1]
        pupil_phase = self.sourceData[2]
        plt.figure(fignum, figsize=[9,3])
        plt.clf()
        plt.subplot(1,3,1)
        plt.imshow(pupil_phase, cmap='twilight')
        plt.title('Pupil phase')
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(psf_ampl**2)
        plt.title('PSF intensity')
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.imshow(psf_phase, cmap='twilight')
        plt.title('PSF phase')
        plt.colorbar()
        plt.pause(0.001)
        # plt.tight_layout()

    def saveAllResults(self, filename='allResults.npz'):
        np.savez(filename, allResults=self.allResults, allFinalFluxVals=self.allFinalFluxVals)

    def saveForNN(self, filename):
        # Slice off piston Zernike
        output_coeffs = [i[1:] for i in self.allCoeffs]
        np.savez(filename, zernikes=output_coeffs, fluxes=self.allFinalFluxVals)

    def loadResults(self, filename='allResults.npz', showPlots=True):
        a = np.load(filename, allow_pickle=True)
        self.allResults=a['allResults']
        self.allFinalFluxVals = a['allFinalFluxVals']

        if showPlots:
            for cur in self.allResults:
                self.FLDampl = cur.FLDampl
                self.FLDintens = cur.FLDintens
                self.FLDphase = cur.FLDphase
                self.MONdata = cur.MONdata
                self.MONposn = cur.MONposn
                self.XZampl = cur.XZampl
                self.YZampl = cur.YZampl
                self.sourceData = cur.sourceData
                self.finalFluxVals = cur.finalFluxVals
                coeffs = cur.coeffs
                # self.allCoeffs += [cur.coeffs]

                self.plotall()
                plt.pause(0.001)
                self.plotSourceData()
                plt.pause(0.001)
                print(coeffs)

            self.plotFinalFluxes()
