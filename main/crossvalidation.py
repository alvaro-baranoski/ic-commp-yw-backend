

def crossvalidation(main_modes, welch_peaks):
    for mode in main_modes:
        for peak in welch_peaks:
            if (mode["freq_interval"][0] <= peak[0] <= mode["freq_interval"][1]):
                mode["crossvalidation"] = True
                break
            else:
                mode["crossvalidation"] = False

    return main_modes