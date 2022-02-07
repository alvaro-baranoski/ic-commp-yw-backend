from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def nan_indexes(nan_array):

	# Get location of nan and not nan values in array as True and False
	nanIndex = np.isnan(nan_array)

	# Get array containing differance values to find consecutive nan numbers
	# Checks for the ocurrance of run of 1's
	nanIndex = np.concatenate(([0], np.equal(nanIndex, True).view(np.int8), [0]))
	nanIndex = np.abs(np.diff(nanIndex))
	nanIndex = np.where(nanIndex == 1)[0]

	return nanIndex


def outlier_removal(outlierArray, k=3):

	# Sets kernel size to be right size and odd
	if len(outlierArray) < 100:
		kernelSize = len(outlierArray) // 10
		if kernelSize % 2 == 0:
			kernelSize += 1
	else:
		kernelSize = 25

	# Calculates median filter and standard deviation
	freqMedianFilter = signal.medfilt(outlierArray, kernel_size=kernelSize)
	freqStdLimit = np.nanstd(outlierArray) * k

	# Saves lower and higher filter limits
	lowerLimit = freqMedianFilter - freqStdLimit
	higherLimit = freqMedianFilter + freqStdLimit

	# Compares limits to data
	freqLowerMask = np.less_equal(outlierArray, lowerLimit)
	freqHigherMask = np.greater_equal(outlierArray, higherLimit)
	freqMask = np.logical_or(freqLowerMask, freqHigherMask)

	# Removes outliers if there's any
	if True in freqMask:
		outlierLocation = np.where(freqMask == True)
		for i in outlierLocation[0]:
			outlierArray[i] = np.nan

	return outlierArray

########################### NIGERIA EXPERIMENT ########################### 

def mean_outlier_removal(outlierArray, k=3):

	# Calculates mean and standard deviation
	freqMean = np.nanmean(outlierArray)
	freqStdLimit = np.nanstd(outlierArray) * k

	# Saves lower and higher filter limits
	lowerLimit = freqMean - freqStdLimit
	higherLimit = freqMean + freqStdLimit

	# Compares limits to data
	freqLowerMask = np.less_equal(outlierArray, lowerLimit)
	freqHigherMask = np.greater_equal(outlierArray, higherLimit)
	freqMask = np.logical_or(freqLowerMask, freqHigherMask)

	# Removes outliers if there's any
	if True in freqMask:
		outlierLocation = np.where(freqMask == True)
		for i in outlierLocation[0]:
			outlierArray[i] = np.nan

	return outlierArray


def find_nan_run(inputArray, run_max=10):
	# Gets array of position of nan values
	nanIndex = nan_indexes(inputArray)

	# Checks indexes of run ocurrances and compare to threshold
	runIndex = 0

	while runIndex < len(nanIndex):
		runSize = nanIndex[runIndex + 1] - nanIndex[runIndex]
		if runSize >= run_max:
			#raise NameError(f"Sequence of {runSize} missing values. That's too many!")
			raise NameError('Muitos dados faltantes em seguida')
		runIndex += 2

	return False


def linear_interpolation(inputArray):
	# Get location of nan and not nan values in array as True and False
	nanLocation = np.isnan(inputArray)
	numberLocation = np.logical_not(nanLocation)

	# Linear interpolation at nan indexes
	inputArray[nanLocation] = np.interp(nanLocation.nonzero()[0],
										numberLocation.nonzero()[0],
										inputArray[~nanLocation])

	return inputArray
