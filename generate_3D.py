import bpy
import numpy as np
import os
import scipy
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import argparse
import sys
from math import pi, atan, acos, sin, cos, sqrt, ceil

M_NUM = 19
F_NUM = 25000
V_NUM = 12500

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

def impute(data, mask, body_data):
    imp = IterativeImputer(max_iter=10, random_state=0)
    if 0 in mask:
        data[~mask] = np.nan
        tmp = body_data.t_measure.copy().transpose()
        imp.fit(tmp)
        data = imp.transform(data.transpose()).transpose()
    return data

def get_output(body_meas, sex="female", impute_verbose=False):
    """
    Get vertices and facets data from 19 inputs
    Inputs should be as-is human readable data.
    Weight is in kilograms (kg), the other 18 are in centimeters (cm).
    Can contain missing values, but weight and height are necessary.
    Missing value should be replaced with 0 beforehand.
    """
    # Make sure body_meas is a numpy array of shape (19,1).
    body_meas = np.array(body_meas).copy().astype(np.float64)
    if body_meas.ndim == 1:
        body_meas = np.expand_dims(body_meas, axis=1)

    assert len(body_meas) == M_NUM
    assert sex in ["male", "female"]
    # Check weight and height existence
    if body_meas[0, 0] == 0 or body_meas[1, 0] == 0:
        print("The weight and height should be given in order to provide accurate rendering.")
        return  
    
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
    filled = impute(filled, mask, body_data)
    # Record human-readable measures after imputation.
    real_human_data = body_data.mean_measure + filled.copy() * body_data.std_measure
    real_human_data[0, 0] = (real_human_data[0, 0]/1000)**3
    real_human_data[1:, 0] /= 10
    # Map to body_mesh data: Vertices, Edges and Faces
    # but we don't need edges.
    filled = np.array(filled).reshape(M_NUM, 1) # Ensure numpy array shape (19,1).
    filled = body_data.m2d.dot(filled) # Now is shape (10,1)
    d = np.matmul(body_data.d_basis, filled) # Now is shape (225000, 1)
    d.shape = (F_NUM, 9) # Reshape to (25000, 9)
    d *= body_data.std_deform
    d += body_data.mean_deform
    d = np.array(d.flat).reshape(d.size, 1)
    Atd = body_data.d2v.transpose().dot(d)
    x = body_data.lu.solve(Atd)
    x = x[:V_NUM*3]
    x.shape = (V_NUM, 3)
    x_mean = np.mean(x, axis=0)
    x -= x_mean
    [v,f] = [x, body_data.facets -1 ]
    v = v.astype('float32')
    v = rotate_body_mesh(v)
    # Shift by nose_y
    nose_y = v[11957][1]
    v[:, 1] -= nose_y
    # Package return values
    v = list(map(tuple, v))
    f = list(map(tuple, f))
    min_z = min([point[2] for point in v])
    return v, f, real_human_data, min_z

def rotate_body_mesh(v):
    R = v[5646][:2]
    L = v[5853][:2]
    if R[0]-L[0] != 0:
        slope = (R[1]-L[1])/(R[0]-L[0])
        theta = pi/2 - atan(slope) 
        Sin = sin(theta)
        Cos = cos(theta)
        rot_matrix = np.array([[Cos, Sin], [-Sin, Cos]])
        v[:, :2]  = np.matmul(v[:, :2], rot_matrix)
        return v
    else: 
        if L[1] < R[1]:
            v[:, :2] *= -1
        return v

def add_material(obj_name):
    obj = bpy.data.objects[obj_name]
    mat = bpy.data.materials.new(name="Color")
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    # mat.diffuse_color = (0.450786, 0.296138, 0.208637, 1)
    # mat.diffuse_color = color
    mat.use_nodes = True
    principled = mat.node_tree.nodes['Principled BSDF']
    # Assign color, specular and roughness to avoid light reflection.
    principled.inputs['Base Color'].default_value = (0.83, 0.83, 0.83, 1)
    principled.inputs[5].default_value = 0
    principled.inputs[7].default_value = 1

if __name__ == '__main__':
    # Change to object mode
    if bpy.context.object is not None:
        if bpy.context.object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
    # Delete all existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Accept parameters. 
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gender", help='gender')
    parser.add_argument("-l", "--list", nargs='+', help='other numeric data (19)')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    body_meas = np.array(list(map(float,args.list)))
    ### Create mesh data from deep learning model. 
    v, f, r, min_z = get_output(body_meas, sex=args.gender)

    # Use mesh data to build 3D-body inside Blender. 
    e = []
    new_mesh = bpy.data.meshes.new("body_mesh")
    new_mesh.from_pydata(v, e, f)
    new_mesh.update()

    new_object = bpy.data.objects.new("body_object", new_mesh)
    view_layer = bpy.context.view_layer
    view_layer.active_layer_collection.collection.objects.link(new_object)

    # Add material
    add_material("body_object")

    # Print imputed measurements
    names = ['Weight', 'Height', 'Neck', 'Chest', 'Belly Button Waist',
            'Gluteal Hip', 'Neck Shoulder Elbow Wrist', 'Crotch Knee Floor', 'Across Back Shoulder Neck', 'Neck to Gluteal Hip',
            'Natural Waist', 'Maximum Hip', 'Natural Waist Raise', 'Shoulder to Midhand', 'Upper Arm',
            'Wrist', 'Outer Natural Waist to Floor', 'Knee', 'Maximum Thigh']
    units = ['kg'] + ['cm']*18
    outputs = {names[i]: f'{round(r[i, 0],2)}{units[i]}' for i in range(19)}
    for key, item in outputs.items():
        print(key, ': ', item)