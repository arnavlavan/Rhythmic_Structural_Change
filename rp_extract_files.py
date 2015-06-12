# RP_EXTRACT_FILES:

# wrapper around rp_extract_python.py to sequentially extract features from all files in a given directory

# 2015-04 by Thomas Lidy

import unicsv # unicode csv library (installed via pip install unicsv)
import time # for time measuring
from audiofile_read import * # reading wav and mp3 files
import rp_extract_python as rp # Rhythm Pattern extractor


def initialize_feature_files(base_filename,ext,append=False):
    files = {}  # files is a dict of one file handle per extension
    writer = {} # files is a dict of one file writer per extension

    if append:
        mode = 'a' # append
    else:
        mode = 'w' # write new (will overwrite)

    for e in ext:
        filename = base_filename + '.' + e
        files[e] = open(filename, mode)
        writer[e] = unicsv.UnicodeCSVWriter(files[e]) #, quoting=csv.QUOTE_ALL)

    return (files,writer)


def write_feature_files(id,feat,writer,id2=None):
    # id: string id (e.g. filename) of extracted file
    # feat: dict containing 1 entry per feature type (must match file extensions)

    for e in feat.keys():
        f=feat[e].tolist()
        f.insert(0,id)        # add filename before vector (to include path, change fil to filename)
        if not id2==None:
            f.insert(1,id2)
        writer[e].writerow(f)


def close_feature_files(files,ext):
    for e in ext:
        files[e].close()


def extract_all_files_in_path(path,out_file,feature_types):

    ext = feature_types

    files, writer = initialize_feature_files(out_file,ext)

    # iterate through all files

    start_abs = time.time()

    n = 0

    # TODO: restrict to *.mp3 (or other extension)

    for d in os.walk(path):    # finds all subdirectories and gets a list of files therein
        subpath = d[0]
        # dir_list = d[1]
        filelist = d[2]
        print subpath, len(filelist), "files"


        for fil in filelist:  # iterate over all files in a dir
            n += 1
            filename = subpath + os.sep + fil
            print '#',n,':', filename

            start = time.time()

            # read audio file (wav or mp3)
            samplerate, samplewidth, data = audiofile_read(filename)

            end = time.time()
            print end - start, "sec"

            # audio file info
            print samplerate, "Hz,", data.shape[1], "channels,", data.shape[0], "samples"

            # extract features
            # Note: the True/False flags are determined by checking if a feature is listed in 'ext' (see settings above)

            start = time.time()

            feat = rp.rp_extract(data,
                              samplerate,
                              extract_rp   = ('rp' in ext),          # extract Rhythm Patterns features
                              extract_ssd  = ('ssd' in ext),           # extract Statistical Spectrum Descriptor
                              extract_sh   = ('sh' in ext),          # extract Statistical Histograms
                              extract_tssd = ('tssd' in ext),          # extract temporal Statistical Spectrum Descriptor
                              extract_rh   = ('rh' in ext),           # extract Rhythm Histogram features
                              extract_trh  = ('trh' in ext),          # extract temporal Rhythm Histogram features
                              extract_mvd  = ('mvd' in ext),        # extract Modulation Frequency Variance Descriptor
                              spectral_masking=True,
                              transform_db=True,
                              transform_phon=True,
                              transform_sone=True,
                              fluctuation_strength_weighting=True,
                              skip_leadin_fadeout=1,
                              step_width=1)

            end = time.time()

            print "Features extracted:", feat.keys(), end - start, "sec"

            #type(feat["rp"])
            #numpy.ndarray

            #print feat["rp"].shape
            #(1440,)

            # WRITE each features to a CSV

            # TODO check if ext and feat.keys are consistent

            start = time.time()

            # id = fil -> add filename before vector (to include path, change fil to filename)
            write_feature_files(fil,feat,writer)

            end = time.time()

            print "Data written." #, end-start

    # close all output files

    close_feature_files(files,ext)

    end = time.time()

    print "FEATURE EXTRACTION FINISHED.", n, "files,", end-start_abs, "sec"



if __name__ == '__main__':

    # SET WHICH FEATURES TO EXTRACT (must be lower case)

    feature_types = ['rp','ssd','rh','mvd'] # sh, tssd, trh

    # SET PATH WITH AUDIO FILES (INPUT)

    in_path = "./music"

    # OUTPUT FEATURE FILES

    out_path = './features'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    out_file = 'features'

    out_filename = out_path + os.sep + out_file

    extract_all_files_in_path(in_path,out_filename,feature_types)
