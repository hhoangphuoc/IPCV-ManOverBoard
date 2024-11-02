# THIS FILE USES TO STORE THE PARAMETERS USING
# FOR THE REST OF THE CODE TO ACCESS
import numpy as np
INITIAL_POSITION = (667, 540)

# Camera parameters
"""
This camera parameters are obtained from the calibration process.
"""
CAMERA_PARAMS = {
    'intrinsic_matrix': np.array([[1.25894746e+03, 0.00000000e+00, 7.14289119e+02],     
       [0.00000000e+00, 1.67617508e+03, 5.54290374e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 
    'distortion_coeffs': np.array([[-1.57545496e-01,  5.36835453e-01,  1.34584590e-03,
         1.13433287e-03, -2.02419879e+00]]), 
    'fx': 1258.9474561730278, 
    'fy': 1676.1750841212438, 
    'cx': 714.2891192943524, 
    'cy': 554.2903742846845
}
#--------------------------------

INTRINSIC_MATRIX = CAMERA_PARAMS['intrinsic_matrix']
DISTORTION_COEFFS = CAMERA_PARAMS['distortion_coeffs']

#--------------------------------


