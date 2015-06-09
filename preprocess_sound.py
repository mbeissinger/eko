import os
import numpy
from feature_helpers import compute_specgram

audio_ext = ['.wav', '.aif']

def main(step, window, max_freq):
    # go through everything in the data/sounds directory
    for f in find_audio_files('data/sounds/'):
        name, ext = os.path.splitext(f)
        processed_name = name + '_processed_%d_%d_%d.npy'%(step, window, max_freq)
        # if the current processed version doesn't exist
        if not os.path.exists(processed_name):
            # process the file!
            processed, _freqs = compute_specgram(audio_file=f, step=step, window=window, max_freq=max_freq)
            if processed is not None:
                print f, '--->', processed_name
                print processed.shape
                numpy.save(processed_name, processed)

def find_audio_files(directory):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            _, ext = os.path.splitext(basename)
            if ext in audio_ext:
                filename = os.path.join(root, basename)
                yield filename

if __name__ == '__main__':
    for freq in [2000, 4000, 8000]:
        for window in [10, 20]:
            main(step=10, window=window, max_freq=freq)