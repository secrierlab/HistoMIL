"""
Utils functions for concepts:

- Contour_Checking functions
"""

##############################################################################
#              Contour_Checking functions from github
###############################################################################

import cv2
import numpy as np

class Contour_Checking_fn(object):
    # Defining __call__ method 
    def __call__(self, pt): 
        raise NotImplementedError

class isInContour_basic(Contour_Checking_fn):
    def __init__(self, contour):
        self.cont = contour

    def __call__(self, pt): 
        return 1 if cv2.pointPolygonTest(self.cont, pt, False) >= 0 else 0

class isInContour_center(Contour_Checking_fn):
    def __init__(self, contour, patch_size):
        self.cont = contour
        self.patch_size = patch_size

    def __call__(self, pt): 
        return 1 if cv2.pointPolygonTest(self.cont, (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2), False) >= 0 else 0


# Easy version of 4pt contour checking function 
# - 1 of 4 points need to be in the contour for test to pass
class isInContour_Easy(Contour_Checking_fn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = (patch_size//2*center_shift)
    def __call__(self, pt): 
        center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
        if self.shift > 0:
            all_points = [(center[0]-self.shift, center[1]-self.shift),
            (center[0]+self.shift, center[1]+self.shift),
            (center[0]+self.shift, center[1]-self.shift),
            (center[0]-self.shift, center[1]+self.shift)
            ]
        else:
            all_points = [center]

        for points in all_points:
            if cv2.pointPolygonTest(self.cont, points, False) >= 0:
                return 1
        return 0

# Hard version of 4pt contour checking function 
# - all 4 points need to be in the contour for test to pass
class isInContour_Hard(Contour_Checking_fn):
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = (patch_size//2*center_shift)
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, points, False) < 0:
				return 0
		return 1

def isInHoles(holes, pt, patch_size):
    for hole in holes:
        if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
            return 1
    
    return 0

def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
    if cont_check_fn(pt):
        if holes is not None:
            return not isInHoles(holes, pt, patch_size)
        else:
            return 1
    return 0

def set_contour_check_fn(contour_fn,contour,ref_patch_size):
    if isinstance(contour_fn, str):
        if contour_fn == 'four_pt':
            cont_check_fn = isInContour_Easy(contour=contour, patch_size=ref_patch_size[0], center_shift=0.5)
        elif contour_fn == 'four_pt_hard':
            cont_check_fn = isInContour_Hard(contour=contour, patch_size=ref_patch_size[0], center_shift=0.5)
        elif contour_fn == 'center':
            cont_check_fn = isInContour_center(contour=contour, patch_size=ref_patch_size[0])
        elif contour_fn == 'basic':
            cont_check_fn = isInContour_basic(contour=contour)
        else:
            raise NotImplementedError
    else:
        assert isinstance(contour_fn, Contour_Checking_fn)
        cont_check_fn = contour_fn
    return cont_check_fn

def scaleContourDim(contours, scale):
    # calc contours after scale
    return [np.array(cont * scale, dtype='int32') for cont in contours]

def scaleHolesDim(contours, scale):
    # calc holes after scale
    return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

def filter_contours( contours, hierarchy, filter_params):
    """
        Filter contours by area.
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
    all_holes = []
    
    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()
        if a == 0: continue
        if tuple((filter_params['a_t'],)) < tuple((a,)): 
            filtered.append(cont_idx)
            all_holes.append(holes)
            
    foreground_contours = [contours[cont_idx] for cont_idx in filtered]
    
    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids ]
        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        # take max_n_holes largest holes by area
        unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
        filtered_holes = []
        
        # filter these holes
        for hole in unfilered_holes:
            if cv2.contourArea(hole) > filter_params['a_h']:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours

