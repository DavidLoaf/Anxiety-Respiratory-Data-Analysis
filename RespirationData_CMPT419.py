import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from joblib import Parallel, delayed
import os

def plotData(cwd, folder_path, file):

    # try: some files ran into errors, it's simpler just to skip them.
    try: 
        # Load file
        mat_data = scipy.io.loadmat(os.path.join(folder_path, file))
        
        # samples -> seconds
        resp_signal = mat_data["data"][:, 1]
        time_axis = np.arange(resp_signal.shape[0]) / 250

        # Downsampling because its taaking wayyyyy too long. 
        loess_downsample = 10
        signal_downsampled = resp_signal[::loess_downsample]
        time_axis_downsampled = time_axis[::loess_downsample]

        
        # loess epicness (thanskads you 353 i love you)
        def loess_chunk(start, end):
            return lowess(signal_downsampled[start:end], time_axis_downsampled[start:end], frac=0.05, return_sorted=False)

        # multithreading because its still taking too long!
        chunk_size = len(time_axis_downsampled) // 4
        chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(4)]

        # Run loess but multithreaded
        smoothed_chunks = Parallel(n_jobs=4)(delayed(loess_chunk)(start, end) for start, end in chunks)

        # Merge smoothed chunks into one array
        smoothed_resp = np.concatenate(smoothed_chunks)

        # Downsample for final plotting 
        downsampled_plot = 10
        time_downsampled = time_axis_downsampled[::downsampled_plot]
        resp_downsampled = smoothed_resp[::downsampled_plot]
        
        # Plot that ong frfr
        plt.figure(figsize=(12, 4))
        plt.plot(time_downsampled, resp_downsampled, color="red", linewidth=0.8)
        plt.ylabel("Amplitude of Respiration")
        plt.xlabel("Time (seconds)")
        plt.title("Respiration Signal (Multi-Threaded LOESS Smoothed)")
        plt.ion()
        plt.show()
        
        filename = file.removesuffix(".mat")
        image_path = os.path.join(cwd, "images", filename)
        
        plt.savefig(f"{image_path}.png")
        plt.close()
        

    except:
        print("error")


def main():

    # get folderpath
    cwd = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(cwd, "data")

    for file in os.listdir(folder_path):

        print(file)

        if file.endswith(".mat"):
            plotData(cwd,folder_path,file)


if __name__ == "__main__":
    main()



