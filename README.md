# Imputation 

This code is an extraction from https://github.com/1900zyh/3D-Human-Body-Shape, which only do imputation for the body measurements. 

### How to use it

After install the requirements, preferrably using Python3.9, run the following code 

```
    python impute.py -g male -l 80 180 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
```

The flag `-g` can only accept `male` and `female`.

The 19 numbers at the end is following this list:

```
'Weight'
'Height'
'Neck'
'Chest'
'Belly Button Waist'
'Gluteal Hip'
'Neck Shoulder Elbow Wrist'
'Crotch Knee Floor'
'Across Back Shoulder Neck'
'Neck to Gluteal Hip'
'Natural Waist'
'Maximum Hip'
'Natural Waist Raise'
'Shoulder to Midhand'
'Upper Arm'
'Wrist'
'Outer Natural Waist to Floor'
'Knee'
'Maximum Thigh'
```
Only 'weight' is in kilograms, others are in centimeters. 

Note: The `0`'s in the command line are the missing values, which will be filled. 