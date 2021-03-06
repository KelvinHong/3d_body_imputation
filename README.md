# Imputation 

This code is an extraction from https://github.com/1900zyh/3D-Human-Body-Shape, which do imputation for the body measurements. We use scikit-learn iterative imputation rather than MICE from fancyimputer, because scikit-learn is a more commonly used library, the result should be similar. 

Update 24 Dec 2021: We will use Blender for 3D-body visualization, which doesn't offer the functionality for user to tweak measurement data in a UI. However, the user can change the measurement data using command line input. 

### To impute missing data

After install the requirements, preferrably using Python3.9, run the following code 

```
    python impute.py -g male -l 80 180 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
```

The flag `-g` can only accept `male` and `female`. 

### To impute missing data and generate 3D body in Blender

Make sure you have Blender (preferably 2.93, but should be fine with any version >=2.80). Run the following code 

```
blender body.blend --python .\generate_3D.py -- -g male -l 80 180 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

This will impute the missing data (those represented by `0`s), print the imputed data, and also open a Blender file showing the 3D-model. 

### Misc

The 19 numbers at the end is following this list:

```
'Weight' 
'Height'
'Neck' - The perimeter around the neck
'Chest' - The perimeter aroudn the chest
'Belly Button Waist' - The perimeter around the waist on Belly Button level
'Gluteal Hip' - The perimeter aroud the hip on the gluteal muscle level
'Neck Shoulder Elbow Wrist' - The length from neck-center to shoulder, to elbow, then to wrist
'Crotch Knee Floor' - The distance from crotch to floor, see https://en.wikipedia.org/wiki/Crotch for definition of crotch
'Across Back Shoulder Neck' - The distance from one shoulder to another, passing the neck
'Neck to Gluteal Hip' - The vertical distance from neck to gluteal muscle
'Natural Waist' - The perimeter aroud the waist
'Maximum Hip' - The largest perimeter around the hip area
'Natural Waist Raise' - The perimeter of the waist, which is near the bottom of chest, a bit higher than natural waist
'Shoulder to Midhand' - The length of shoulder to elbow, to wrist, to midhand
'Upper Arm' - The length of arm from shoulder to elbow
'Wrist' - The wrist size
'Outer Natural Waist to Floor' The vertical distance from waist to floor
'Knee' - (Not sure)
'Maximum Thigh' - The maximum perimeter of thigh of one leg. 
```
Only 'weight' is in kilograms, others are in centimeters. 
The meaning of above names are not 100% accurate, feel free to suggest us the accurate meaning of them. 

Note: The `0`'s in the command line are the missing values, which will be filled. 