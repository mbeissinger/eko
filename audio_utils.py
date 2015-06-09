''' Collection of classes to help with finding and adding noise to audio
'''

import os
import wave
import pydub
import random
import string
import subprocess
import numpy as np
import glob
from scipy import stats
from cStringIO import StringIO
# from utils import parmap
import scikits.audiolab as ab
import linecache
import re
import json

MSEC = 1000 # pydub frames are in ms, so multiply with MSEC when indexing
FRAME_RATE = 16000
DEVNULL = open(os.devnull, 'w')
ROOT_DIR = '/scratch/sanjeev/random-youtube-1/'
TMP_DIR = '/scratch/sanjeev/tmp/'

class AudioSegment(pydub.AudioSegment):
    ''' Helper class to manipulate `AudioSegment`s
        Some functions may use sox
    '''
    sox_cmd = 'sox -t wav - -t wav -'
    effects = ['echo', 'chorus', 'speed', 'pitch', 'flanger', 'reverb', 'tempo', 'overdrive']

    @property
    def power_dB(self):
        ''' power of the segment in decibels
        '''
        if self.rms == 0:
            return 0
        return pydub.utils.ratio_to_db(self.rms)

    def normalize(self, required_power_in_dB=70):
        ''' Performs a best effort power normalization
            If the power could not be set to the required_power_in_dB
            it returns the altered audio without failing!
        '''
        delta = required_power_in_dB - self.power_dB
        self.normalize_gap = delta
        self = self + delta
        self = self.set_frame_rate(FRAME_RATE)
        self.__class__ = AudioSegment
        self.normalize_gap = abs(self.power_dB - required_power_in_dB) < 5
        return self

    @property
    def wav_data(self):
        assert self.frame_rate == FRAME_RATE, "Frame rate set incorrectly."
        assert hasattr(self, 'normalize_gap'), "Normalize audio before accessing wav_data."
        wav_file = StringIO()
        self.export(wav_file, format='wav')
        wav_file.seek(0)
        return wav_file

    @classmethod
    def get_random_effect(cls):
        return cls.effects[random.randint(0, len(cls.effects)-1)]

    def add_effect(self, effect, params=[]):
        ''' Uses sox to add one of these effects
            Read AudioSegment.effects to see the list of effects available
        '''
        # Write the current file to console
        assert self.duration_seconds < 100, '{0:.2f}'.format(self.duration_seconds)
        original_file = StringIO()
        self.export(original_file, format='wav')
        original_file.seek(0)
        wav_data = original_file.read()

        # Add the effect using sox
        try:
            effect_cmd = AudioSegment._get_effect_param(effect, params)
            sox_cmd = '{0} {1}'.format(self.sox_cmd, effect_cmd)
            process = subprocess.Popen(sox_cmd, stdout=subprocess.PIPE,
                                       stdin=subprocess.PIPE, stderr=DEVNULL,
                                       shell=True)
            effected_file = StringIO(process.communicate(wav_data)[0])
            effected_file.seek(0)

            # Return the updated one
            return AudioSegment.from_wav(effected_file)
        except Exception as msg:
            print msg, effect
            return self

    @classmethod
    def _get_effect_param(cls, effect, params=[]):
        # Make sure params is a list
        if not hasattr(params, '__getitem__'): params = [params]


        if effect == 'echo':
            delay = random.randint(50, 350)
            decay = random.randint(1, 7) * 0.1
            return 'echo 1 1 {0} {1:.2}'.format(delay, decay)

        elif effect == 'chorus':
            opts = 'chorus 1 1 '
            num_people = random.randint(1, 5)
            for i in range(num_people):
                delay = random.randint(21, 90)
                decay = random.randint(1, 7) * 0.1
                speed = random.random() + 0.5
                depth = random.random() + 1.5
                opts += ' {0} {1:.2} {2:.2} {3:.2} -s' \
                        .format(delay, decay, speed, depth)
            return opts

        elif effect == 'flanger':
            delay = random.randint(0, 30)
            depth = random.randint(0, 10)
            regen = random.randint(-95, 95)
            width = random.randint(60, 100)
            speed = random.random() + 0.1
            phase = random.randint(0, 100)
            return 'flanger {0} {1} {2} {3} {4:.2} sin {5} lin' \
                    .format(delay, depth, regen, width, speed, phase)

        elif effect == 'reverb':
            return 'reverb '

        elif effect == 'tempo':
            amount = params[0] if len(params) > 0 else random.random() + 0.5
            return 'tempo -s {0:.2}'.format(amount)

        elif effect == 'speed':
            amount = params[0] if len(params) > 0 else random.random() + 0.5
            return 'speed {0:.2}'.format(amount)

        elif effect == 'pitch':
            amount = params[0] if len(params) > 0 else random.randint(-400, 400)
            return 'pitch {0}'.format(amount)

        elif effect == 'overdrive':
            amount = params[0] if len(params) > 0 else 20*abs(random.random())
            return 'overdrive {0:.2}'.format(amount)


        return ''

    def sample_clip(self, required_duration):
        ''' Randomly sample a clip of `required_duration` seconds from self
        '''
        self_duration = int(self.duration_seconds * MSEC) # in ms
        required_duration = int(required_duration * MSEC) # in ms
        assert self_duration >= 20 + required_duration, 'Cannot sample!'
        start_time = random.randint(10, self_duration-required_duration-10)
        clip = self[start_time:start_time+required_duration]
        clip.__class__ = AudioSegment
        return clip

    def add_noise(self, noise, snr_dB, final_duration=None, effect=None):
        ''' Adds the given noise signal such that the snr_dB is maintained
        '''
        signal_duration = int(self.duration_seconds * MSEC) # in ms
        noise_duration = int(noise.duration_seconds * MSEC) # in ms
        assert noise_duration >= signal_duration, 'Noise smaller than signal'
        if effect:
            noise = noise.add_effect(effect)
        if final_duration:
            noise = noise.sample_clip(final_duration or self.duration_seconds)
        delta = self.power_dB - noise.power_dB - snr_dB
        noise_clip = noise + delta
        merged = noise_clip * self
        merged.__class__ = AudioSegment
        return merged

    def make_segments(self, prefix, duration=60, outdir=TMP_DIR):
        ''' Saves consecutive `duration` seconds long clips from self into
            outdir with names of form `prefix.{starting_second}.wav`
        '''
        split_files = []
        duration = int(duration * MSEC)
        for start_pos in range(0, len(self), duration):
            seg_x = self[start_pos:start_pos + duration]
            if seg_x.duration_seconds * MSEC >= duration:
                basename = '{0}.{1}.wav'.format(prefix, start_pos)
                fname = os.path.join(outdir, basename)
                seg_x.set_channels(1)
                seg_x.set_frame_rate(FRAME_RATE)
                seg_x.export(fname, format='wav')
                split_files.append((basename, duration))
        return split_files

class Youtube(object):
    ''' Pulls and converts audio from youtube '''

    url_alpha = list(string.ascii_letters + string.digits)
    base_url = 'https://www.youtube.com/watch?v=%s'

    ydl_cmd = ['youtube-dl', '-x', '--cache-dir', TMP_DIR, '--output']

    avconv_cmd = 'avconv -y -f {2} -i {0} -vn -b 64k -ar {3} -ac 1 -f flac {1}'
    ydl_filename_format = '%(id)s.%(ext)s'

    def __init__(self, yt_id, deloop=False):
        ''' Initializing this class downloads a youtube file in wav format
            into `ROOT_DIR`
        '''
        self.yt_id = yt_id
        url = self.base_url % self.yt_id
        audio_fname = os.path.join(TMP_DIR, self.ydl_filename_format)
        ydl_params = self.ydl_cmd + [audio_fname, url]
        tmp_file = glob.glob(audio_fname % {'id': yt_id, 'ext' : '*'})
        final_fname = os.path.join(ROOT_DIR, self.ydl_filename_format)
        sound_file = final_fname % {'id': yt_id, 'ext': 'wav'}
        if os.path.exists(sound_file):
            print sound_file, 'exists'
            return

        if len(tmp_file) == 0:
            status = subprocess.call(ydl_params, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=DEVNULL)
            if status != 0:
                print 'downloading ', yt_id, ' failed'
                return

        tmp_file = glob.glob(audio_fname % {'id': yt_id, 'ext' : '*'})[0]
        inpformat = tmp_file[tmp_file.rindex('.')+1:]

        if deloop:
            # The headers are wrong in long files, use flac
            # for all processing on large files
            sound_file = self._flac_encode(tmp_file)
            self.audio = self.get_loopless_segment(sound_file)
        else:
            self.audio = AudioSegment.from_file(tmp_file, format=inpformat)

        print 'saving to', sound_file
        self.audio.set_channels(1)
        self.audio.set_frame_rate(FRAME_RATE)
        # TODO Best if the bit rate could also be set to 64k,
        # usually gives about 2x savings in disk space consumed
        self.audio.export(sound_file, format='wav')

    def _random_yt_id(self):
        ''' Get a random youtube video id '''
        return random.sample(self.url_alpha, 11)

    def _flac_encode(self, inpfile):
        ''' Youtube sounds are downloaded as m4a files. These cannot be loaded
            by scikit.audiolab. Wav files do not support files > 2G,
            so we temporarily convert to a flac format with smaller sample and
            bit rate
        '''
        inpformat = inpfile[inpfile.rindex('.')+1:]
        flacfile = inpfile.replace(inpformat, 'flac')
        conv_cmd = self.avconv_cmd.format(inpfile, flacfile, inpformat, FRAME_RATE)
        subprocess.call(conv_cmd.split(' '), stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=DEVNULL)
        os.remove(inpfile)
        return flacfile

    def get_loopless_segment(self, inpfile):
        inpformat = inpfile[inpfile.rindex('.')+1:]
        corrfname = inpfile.replace(inpformat, 'corr.npy')
        if not os.path.exists(corrfname):
            print corrfname, 'not found'
            source = ab.Sndfile(inpfile, 'r')
            source_data = source.read_frames(source.nframes)
            num_frames = source_data.shape[0]
            clip_size = int(0.1 * FRAME_RATE)
            max_start_pos = FRAME_RATE * 10
            start_pos = random.randint(0, num_frames-clip_size-10)
            start_pos = min(max_start_pos, start_pos)
            clip_data = source_data[start_pos:start_pos+clip_size]
            corr = norm_correlation(clip_data, source_data)
            # Why are there duplicates!
            with open(corrfname, 'w') as save_file:
                np.save(save_file, corr)
        else:
            corr = np.load(corrfname)

        reps = np.sort(np.unique(np.where(corr > 0.98)[0]))
        gaps = np.array([])

        if reps.shape[0] > 2:
            gaps = reps[1:] - reps[:-1]
            gaps = gaps[np.unique(np.where(gaps > 10.)[0])]

        audio = AudioSegment.from_file(inpfile, inpformat)
        os.remove(inpfile)
        file_duration = audio.duration_seconds / 60.
        if gaps.shape[0] > 0:
            loop_duration = stats.mode(gaps)[0][0]/FRAME_RATE
            audio = audio[0:loop_duration * MSEC]
            print '{0} {1:.2f} {2:.2f}'.format(inpfile, file_duration,
                                               audio.duration_seconds / 60.)
        return audio

def norm_correlation(t, s):
    ''' Fast normalized correlation function for matching template `t` against
        the source `s`
        Refer http://stackoverflow.com/questions/23705107
    '''
    import pandas as pd
    n = len(t)
    nt = (t-np.mean(t))/(np.std(t)*n)
    sum_nt = nt.sum()
    a = pd.rolling_mean(s, n)[n-1:-1]
    b = pd.rolling_std(s, n)[n-1:-1]
    b *= np.sqrt((n-1.0) / n)
    c = np.convolve(nt[::-1], s, mode="valid")[:-1]
    result = (c - sum_nt * a) / b
    return result

def get_pcm_audio(pcmdata, nchannel=1, samplerate=8000, nbit=16):
    ''' Converts pcm data to an AudioSegment '''
    wavbuffer = StringIO()
    wavfile = wave.open(wavbuffer, 'wb')
    wavfile.setparams((nchannel, nbit/8, samplerate, 0, 'NONE', 'NONE'))
    wavfile.writeframes(pcmdata)
    wavfile.close()

    wavbuffer.seek(0)
    return AudioSegment.from_wav(wavbuffer)


current_seq_filename = "" #Need filename to know when to clear cache
def getline(seq_filename, line_num):
    '''Gets line of text.  Clears linecache if file changes'''
    global current_seq_filename

    if seq_filename != current_seq_filename:
        linecache.clearcache()
        current_seq_filename = seq_filename

    return linecache.getline(seq_filename, line_num)

def get_raw_audio(fname):
    ''' Loads audio segment from file. Load method depends on file extension '''

    audio_fmt = fname[1+fname.rindex('.'):]
    if audio_fmt == 'pcm':
        with open(fname, 'rb') as pcmfile:
            pcmdata = pcmfile.read()
        audio = get_pcm_audio(pcmdata)

    elif re.match("""seq_\d+""", audio_fmt):
        pos = fname.rindex('_')
        seq_filename, line_num = fname[:pos], int(fname[1+pos:])

        data_line = getline(seq_filename, line_num)
        audio_data = json.loads(data_line)["audio_data"].decode('base64')

        try:
            # Assume this is a wav file
            audio = AudioSegment(data=audio_data)
        except Exception as e:
            # if not, its pcm. We only have these two.
            audio = get_pcm_audio(audio_data)
    else:
        audio = AudioSegment.from_file(fname, format=audio_fmt)

    return audio

def get_wav_duration(wavfile):
    return AudioSegment.from_wav(wavfile).duration_seconds

def download_sounds(ytid):
    try:
        ytid = ytid.strip()
        if ytid:
            Youtube(ytid)
    except Exception as error:
        print ytid, error

# if __name__ == '__main__':
#     with open('ytids', 'r') as yt:
#         ids = yt.readlines()
#     parmap(download_sounds, ids)