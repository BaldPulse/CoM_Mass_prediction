import os
import numpy as np

def load_data(dir):
    data_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npz')]
    print("total data files: ", len(data_files))
    PC_data = []
    target_values = []
    for f in data_files:
        data = np.load(f)
        # print(data['pc'].shape)
        # print(data['target_values'].shape)
        PC_data.append(data['pc'])
        target_values.append(data['target_values'])
    print("PC_data: ", len(PC_data))
    print("target_values: ", len(target_values))
    return np.array(PC_data), np.array(target_values)

def get_PC_stats(dir):
    # try to find the mean and std of the point clouds, if it exists in an npy file
    try:
        mean_std = np.load(os.path.join(dir, 'mean_std.npy'))
        pc_mean = mean_std[0]
        pc_std = mean_std[1]
    except:
        # if the file doesn't exist, calculate the mean and std of the point clouds
        PC_data, target_values = load_data(dir)
        pc_mean = np.mean(PC_data, axis=(0,1))
        pc_std = np.std(PC_data, axis=(0,1))
        np.save(os.path.join(dir, 'mean_std.npy'), [pc_mean, pc_std])
    return pc_mean, pc_std


def normalize_data(PC_data, target_values, pc_mean, pc_std):
    # normalize the point cloud
    Norm_PC = (PC_data - pc_mean) / pc_std
    # normalize the first 3 columns of the target values
    Norm_Target_Values = np.zeros(target_values.shape)
    Norm_Target_Values[:, :, :3] = (target_values[:, :, :3] - pc_mean) / pc_std
    Norm_Target_Values[:, :, 3] = target_values[:, :, 3]
    return Norm_PC, Norm_Target_Values