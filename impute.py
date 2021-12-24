
import numpy as np
import os
import scipy
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import argparse
import sys
import pickle


M_NUM = 19

class Reshaper:
    def __init__(self, label="female"):
        model_path = './release_model/'
        self.label_ = label
        self.facets = np.load(open(os.path.join(model_path, "facets.npy"), "rb"))
        self.m2d = np.load(open(os.path.join(model_path, "%s_m2d.npy"%label), "rb"))
        self.d_basis = np.load(open(os.path.join(model_path, "%s_d_basis.npy"%label), "rb"))
        self.t_measure = np.load(open(os.path.join(model_path, "%s_t_measure.npy"%label), "rb"))
        self.mean_measure = np.load(open(os.path.join(model_path, "%s_mean_measure.npy"%label), "rb"))
        self.mean_deform = np.load(open(os.path.join(model_path, "%s_mean_deform.npy"%label), "rb"))
        self.std_measure = np.load(open(os.path.join(model_path, "%s_std_measure.npy"%label), "rb"))
        self.std_deform = np.load(open(os.path.join(model_path, "%s_std_deform.npy"%label), "rb"))

        loader = np.load(os.path.join(model_path, "%s_d2v.npz"%label))
        self.d2v = scipy.sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])),shape=loader['shape'])
        self.lu = scipy.sparse.linalg.splu(self.d2v.transpose().dot(self.d2v).tocsc())


def impute_wrap(body_meas, sex):
    assert len(body_meas) == M_NUM
    assert sex in ["male", "female"]
    # Check weight and height existence
    if body_meas[0, 0] == 0 or body_meas[1, 0] == 0:
        print("The weight and height should be given in order to provide accurate rendering.")
        
    
    # Preparing data
    body_data = Reshaper(label=sex)
    # Encode data into homogeneous form.  
    encoded = body_meas
    mask = np.zeros((M_NUM, 1), dtype=bool)
    encoded[0, 0] = 1000* (encoded[0, 0]**(1.0/3.0))
    mask[0,0] = 1
    for i in range(1,M_NUM):
        encoded[i, 0] = encoded[i, 0] * 10
        if encoded[i, 0] != 0:
            mask[i, 0] = 1
        else:
            mask[i, 0] = 0
    # Normalize data based on existing datas.
    normed = encoded
    for i in range(M_NUM):
        if normed[i,0] != 0:
            normed[i,0] = (normed[i,0] - body_data.mean_measure[i,0]) / body_data.std_measure[i,0]
    
    # Imputation start
    filled = normed.copy()
    imp = IterativeImputer(max_iter=10, random_state=0)
    if 0 in mask:
        filled[~mask] = np.nan
        tmp = body_data.t_measure.copy().transpose()
        imp.fit(tmp)
        filled = imp.transform(filled.transpose()).transpose()
    # filled = impute(filled, mask, body_data)
    # Record human-readable measures after imputation.
    real_human_data = body_data.mean_measure + filled.copy() * body_data.std_measure
    real_human_data[0, 0] = (real_human_data[0, 0]/1000)**3
    real_human_data[1:, 0] /= 10

    return real_human_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gender", help='gender')
    parser.add_argument("-l", "--list", nargs='+', help='other numeric data (19)')
    args = parser.parse_args()
    body_meas = np.array(list(map(float,args.list)))
    # Make sure body_meas is a numpy array of shape (19,1).
    body_meas = np.array(body_meas).copy().astype(np.float64)
    if body_meas.ndim == 1:
        body_meas = np.expand_dims(body_meas, axis=1)
    sex = args.gender

    r = impute_wrap(body_meas, sex)
    names = ['Weight', 'Height', 'Neck', 'Chest', 'Belly Button Waist',
            'Gluteal Hip', 'Neck Shoulder Elbow Wrist', 'Crotch Knee Floor', 'Across Back Shoulder Neck', 'Neck to Gluteal Hip',
            'Natural Waist', 'Maximum Hip', 'Natural Waist Raise', 'Shoulder to Midhand', 'Upper Arm',
            'Wrist', 'Outer Natural Waist to Floor', 'Knee', 'Maximum Thigh']
    units = ['kg'] + ['cm']*18
    outputs = {names[i]: f'{round(r[i, 0],2)}{units[i]}' for i in range(19)}
    for key, item in outputs.items():
        print(key, ': ', item)
    
    
