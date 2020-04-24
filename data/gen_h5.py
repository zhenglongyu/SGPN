import os
import numpy as np
import data_prep_util
import indoor3d_util
import glob

# Constants
ToothDataPath = 'ToothData'
NUM_POINT = 4096
data_dtype = 'float32'
label_dtype = 'int32'

# Set paths
# filelist = os.path.join(ToothDataPath, 'meta/areaexcept5_data_label.txt')
# data_label_files = [os.path.join(ToothDataPath, 'annotation/', line.rstrip()) for line in open(filelist)]
data_label_files=glob.glob(os.path.join(ToothDataPath,'txt/*.txt'))
output_dir = 'tooth_ins_seg_hdf5'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_room_filelist = os.path.join(output_dir, 'tooth_filelist.txt')
fout_tooth = open(output_room_filelist, 'w')

sample_cnt = 0
data_origin=[]
for i in range(0, len(data_label_files)):
    data_label_filename = data_label_files[i]
    fname = os.path.basename(data_label_filename).strip('.txt')
    # if not os.path.exists(data_label_filename):
    #     continue
    [data, label, inslabel], data_origin = indoor3d_util.tooth2blocks_wrapper_normalized(data_label_filename, data_origin, NUM_POINT, block_size=1.0, stride=0.5,
                                                 random_sample=False, sample_num=None)
    for _ in range(data.shape[0]):
        fout_tooth.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    h5_filename = os.path.join(output_dir, '%s.h5' % fname)
    print('{0}: {1}, {2}, {3}'.format(h5_filename, data.shape, label.shape, inslabel.shape))
    data_prep_util.save_h5ins(h5_filename,
                              data,
                              label,
                              inslabel,
                              data_dtype, label_dtype)

fout_tooth.close()
print("Total samples: {0}".format(sample_cnt))
