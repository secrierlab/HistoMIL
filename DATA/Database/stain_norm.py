"""
For stain normalisation,inspired by StainTools and tiatoolbox.
[https://github.com/Peter554/StainTools]
[https://github.com/TissueImageAnalytics/tiatoolbox/]

"""
from sklearn.decomposition import DictionaryLearning
import numpy as np
import numpy.random as random
import cv2 
from PIL import Image, ImageFilter
import skimage
from skimage import color,exposure





class StainNormalizer:
    """Stain normalization base class.

    """

    def __init__(self):
        self.worker=None
        self.method_name = None

    def set_worker(self,method_name):
        assert method_name in ["reuifrok","macenko", "vahadane","custom","reinhard"]
        if method_name == "reinhard":
            self.worker = StainWorker_Reinhard(method_name=method_name)
        else:
            self.worker = StainWorker_M(method_name=method_name)


    def fit(self, target,paras={}):
        # fit to get normalisation paras
        self.worker.fit(target=target,paras=paras)

    def transform(self,img,paras={}):
        # transform img to target
        return self.worker.transform(img=img,paras=paras)

##############################################################
#            abstract class
###############################################################
class StainWorker:

    def fit(self,target, paras:dict={}):
        raise NotImplementedError

    def transform(self, img, paras:dict={}):
        raise NotImplementedError


##############################################################
#            worker class
###############################################################


class StainWorker_M(StainWorker):
    """
    for the stain normaliser worked with stain matrix
    """
    def __init__(self,method_name):
        self.method_name = method_name
        self.method_fn = {"reuifrok":self.reuifrok_M,
                            "macenko":self.macenko_M,
                            "vahadane":self.vahadane_M,
                            "custom":None,}

        self.stain_M = None
        self.targetC = None
        self.max_targetC = None
        self.RGB_M = None

    def fit(self, target:np.ndarray, paras:dict={}):

        paras.update({"img":target})
        # Fit to a target image.
        self.stain_M = self.get_stain_M(paras)
        self.target_C = self.get_C(target, self.stain_M)

        self.max_targetC = np.percentile(self.target_C, 99, axis=0).reshape((1, 2))
        # useful to visualize.
        self.RGB_M = od2rgb(self.stain_M)

    def transform(self, img:np.ndarray, paras:dict={}):
        paras.update({"img":img})
        #Transform an image to RGB stain normalized image.

        M = self.get_stain_M(paras)

        source_C = self.get_C(img, M)

        maxC_source = np.percentile(source_C, 99, axis=0).reshape((1, 2))
        source_C *= self.max_targetC / maxC_source

        trans = 255 * np.exp(-1 * np.dot(source_C, self.stain_M))

        # ensure between 0 and 255
        trans = self.within_range(trans,min_v=0,max_v=255)

        return trans.reshape(img.shape).astype(np.uint8)

    def get_stain_M(self,paras:dict=None,calc_fn=None,):
        assert self.method_name in self.method_fn.keys()

        if self.method_name == "custom":
            self.method_fn["custom"] = calc_fn

        M = self.method_fn[self.method_name](paras)

        self.check_stain_M(M)
        return M

    def get_C(self,img, stain_M):
        #Estimate concentration matrix given an image and stain matrix.
        optical_density = rgb2od(img).reshape((-1, 3))
        x, _, _, _ = np.linalg.lstsq(stain_M.T, optical_density.T, rcond=-1)
        return x.T

    def within_range(self,img,min_v=0,max_v=255):
        # confirm an img matrix within range
        
        # ensure between min_v and max_v
        img[img > 255] = 255
        img[img < 0] = 0

        return img

    def check_stain_M(self,M):
        if M.shape not in [(2, 3), (3, 3)]:
            raise ValueError("Stain matrix must be either (2,3) or (3,3)")
    
    def reuifrok_M(self,paras:dict):
        return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])

    def macenko_M(self,paras:dict):
        """Macenko stain extractor.
        Get the stain matrix as defined in:
        Macenko, Marc, et al. "A method for normalizing histology
        slides for quantitative analysis." 2009 IEEE International
        Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.
        This class contains code inspired by StainTools
        [https://github.com/Peter554/StainTools] written by Peter Byfield.
        """
        img = paras["img"].astype("uint8")  # ensure input image is uint8
        luminosity_threshold=paras["luminosity_threshold"] if ("luminosity_threshold" in paras.keys()) else 0.8 
        angular_percentile=paras["angular_percentile"] if ("angular_percentile" in paras.keys()) else 99 

        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(img, threshold=luminosity_threshold).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # eigenvectors of cov in OD space (orthogonal as cov symmetric)
        _, e_vects = np.linalg.eigh(np.cov(img_od, rowvar=False))

        # the two principle eigenvectors
        e_vects = e_vects[:, [2, 1]]

        # make sure vectors are pointing the right way
        e_vects = vectors_in_correct_direction(e_vectors=e_vects)

        # project on this basis.
        proj = np.dot(img_od, e_vects)

        # angular coordinates with repect to the prinicple, orthogonal eigenvectors
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        # min and max angles
        min_phi = np.percentile(phi, 100 - angular_percentile)
        max_phi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(e_vects, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = np.dot(e_vects, np.array([np.cos(max_phi), np.sin(max_phi)]))

        # order of H&E - H first row
        he = h_and_e_in_right_order(v1, v2)

        M = he / np.linalg.norm(he, axis=1)[:, None]
        return M

    def vahadane_M(self,paras:dict):
        """
        Vahadane stain extractor.
        Get the stain matrix as defined in:
        Vahadane, Abhishek, et al. "Structure-preserving color normalization
        and sparse stain separation for histological images."
        IEEE transactions on medical imaging 35.8 (2016): 1962-1971.
        """
        img = paras["img"].astype("uint8")  # ensure input image is uint8
        luminosity_threshold=paras["luminosity_threshold"] if ("luminosity_threshold" in paras.keys()) else 0.8 
        regulariser=paras["regulariser"] if ("regulariser" in paras.keys()) else 0.1

        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(
            img, threshold=luminosity_threshold
        ).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # do the dictionary learning
        dl = DictionaryLearning(
            n_components=2,
            alpha=regulariser,
            transform_alpha=regulariser,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            positive_dict=True,
            verbose=False,
            max_iter=3,
            transform_max_iter=1000,
        )
        dictionary = dl.fit_transform(X=img_od.T).T

        # order H and E.
        # H on first row.
        dictionary = dl_output_for_h_and_e(dictionary)

        normalized_rows = dictionary / np.linalg.norm(dictionary, axis=1)[:, None]

        return normalized_rows

###############################################################
#            utils functions for M workers
###############################################################

def rgb2od(img):
    """Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).
    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`): Image RGB
    Returns:
        :class:`numpy.ndarray`: Optical denisty RGB image.
    """
    img = img.copy()
    mask = img == 0
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def od2rgb(OD):
    """Convert from optical density (OD_RGB) to RGB.
    RGB = 255 * exp(-1*OD_RGB)
    Args:
        OD (:class:`numpy.ndarray`): Optical denisty RGB image
    Returns:
        numpy.ndarray: Image RGB
    """
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def vectors_in_correct_direction(e_vectors):
    """Points the eigen vectors in the right direction.
    Args:
        e_vectors (:class:`numpy.ndarray`): eigen vectors
    Returns:
        ndarray pointing in the correct direction
    """
    if e_vectors[0, 0] < 0:
        e_vectors[:, 0] *= -1
    if e_vectors[0, 1] < 0:
        e_vectors[:, 1] *= -1

    return e_vectors


def h_and_e_in_right_order(v1, v2):
    """Rearranges input vectors for H&E in correct order with H as first output.
    Args:
        v1 (:class:`numpy.ndarray`): Input vector for stain extraction.
        v2 (:class:`numpy.ndarray`): Input vector for stain extraction.
    Returns:
        input vectors in the correct order.
    """
    if v1[0] > v2[0]:
        he = np.array([v1, v2])
    else:
        he = np.array([v2, v1])

    return he


def dl_output_for_h_and_e(dictionary):
    """Return correct value for H and E from dictionary learning output.
    Args:
        dictionary (:class:`numpy.ndarray`):
        :class:`sklearn.decomposition.DictionaryLearning` output
    Returns:
        ndarray with correct values for H and E
    """
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]

    return dictionary

def contrast_enhancer(img, low_p=2, high_p=98):
    """Enhancing contrast of the input image using intensity adjustment.
       This method uses both image low and high percentiles.
    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
            Image should be uint8.
        low_p (scalar): low percentile of image values to be saturated to 0.
        high_p (scalar): high percentile of image values to be saturated to 255.
            high_p should always be greater than low_p.
    Returns:
        img (:class:`numpy.ndarray`): Image (uint8) with contrast enhanced.
    Raises:
        AssertionError: Internal errors due to invalid img type.
    """
    # check if image is not uint8
    if not img.dtype == np.uint8:
        raise AssertionError("Image should be uint8.")
    img_out = img.copy()
    p_low, p_high = np.percentile(img_out, (low_p, high_p))
    if p_low >= p_high:
        p_low, p_high = np.min(img_out), np.max(img_out)
    if p_high > p_low:
        img_out = exposure.rescale_intensity(
            img_out, in_range=(p_low, p_high), out_range=(0.0, 255.0)
        )
    return np.uint8(img_out)

def get_luminosity_tissue_mask(img, threshold):
    """Get tissue mask based on the luminosity of the input image.
    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
        threshold (float): luminosity threshold used to determine tissue area.
    Returns:
        tissue_mask (:class:`numpy.ndarray`): binary tissue mask.
    """
    img = img.astype("uint8")  # ensure input image is uint8
    img = contrast_enhancer(img, low_p=2, high_p=98)  # Contrast  enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    if tissue_mask.sum() == 0:
        raise ValueError("Empty tissue mask computed")

    return tissue_mask

###############################################################
#            worker class for Reinhard method
###############################################################
class StainWorker_Reinhard(StainWorker):
    def __init__(self,method_name):
        self.method_name = "reinhard"

        self.target_means = None
        self.target_stds = None

    def fit(self,target,paras:dict={}):
        self.target_means,self.target_stds =self.get_mean_std(img = target)

    def transform(self,img,paras:dict={}):
        chan1, chan2, chan3 = self.rgb2lab_split(img)
        means, stds = self.get_mean_std(img)
        norm1 = ((chan1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((chan2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((chan3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return self.lab_split2rgb(norm1, norm2, norm3)

    def get_mean_std(self, img):
        """Get mean and standard deviation of each channel.
        """
        img = img.astype("uint8")  # ensure input image is uint8
        chan1, chan2, chan3 = self.rgb2lab_split(img)
        m1, sd1 = cv2.meanStdDev(chan1)
        m2, sd2 = cv2.meanStdDev(chan2)
        m3, sd3 = cv2.meanStdDev(chan3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds

    def rgb2lab_split(self,img):
        """Convert from RGB uint8 to LAB and split into channels.
        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`): Input image.
        Returns:
            chan1 (float): L.
            chan2 (float): A.
            chan3 (float): B.
        """
        img = img.astype("uint8")  # ensure input image is uint8
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_float = img.astype(np.float32)
        chan1, chan2, chan3 = cv2.split(img_float)
        chan1 /= 2.55  # should now be in range [0,100]
        chan2 -= 128.0  # should now be in range [-127,127]
        chan3 -= 128.0  # should now be in range [-127,127]
        return chan1, chan2, chan3

    def lab_split2rgb(self,chan1, chan2, chan3):
        """Take seperate LAB channels and merge back to give RGB uint8.
        Args:
            chan1 (float): L channel.
            chan2 (float): A channel.
            chan3 (float): B channel.
        Returns:
            ndarray uint8: merged image.
        """
        chan1 *= 2.55  # should now be in range [0,255]
        chan2 += 128.0  # should now be in range [0,255]
        chan3 += 128.0  # should now be in range [0,255]
        img = np.clip(cv2.merge((chan1, chan2, chan3)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)



######################################################################
def naive_normalize(image):
    """A naive Normalizing function """
    target = np.array([[57.4, 15.84], [39.9, 9.14], [-22.34, 6.58]])

    whitemask = color.rgb2gray(image)
    whitemask = whitemask > (215 / 255)

    imagelab = color.rgb2lab(image)

    imageL, imageA, imageB = [imagelab[:, :, i] for i in range(3)]

    # mask is valid when true
    imageLM = np.ma.MaskedArray(imageL, whitemask)
    imageAM = np.ma.MaskedArray(imageA, whitemask)
    imageBM = np.ma.MaskedArray(imageB, whitemask)

    ## Sometimes STD is near 0, or 0; add epsilon to avoid div by 0 -NI
    epsilon = 1e-11

    imageLMean = imageLM.mean()
    imageLSTD = imageLM.std() + epsilon

    imageAMean = imageAM.mean()
    imageASTD = imageAM.std() + epsilon

    imageBMean = imageBM.mean()
    imageBSTD = imageBM.std() + epsilon

    # normalization in lab
    imageL = (imageL - imageLMean) / imageLSTD * target[0][1] + target[0][0]
    imageA = (imageA - imageAMean) / imageASTD * target[1][1] + target[1][0]
    imageB = (imageB - imageBMean) / imageBSTD * target[2][1] + target[2][0]

    imagelab = np.zeros(image.shape)
    # clip to confirm within colour range
    imagelab[:, :, 0] = np.clip(imageL,0,100)
    imagelab[:, :, 1] = np.clip(imageA,-127,128)
    imagelab[:, :, 2] = np.clip(imageB,-128,127)

    # Back to RGB space
    returnimage = color.lab2rgb(imagelab)
    returnimage = np.clip(returnimage, 0, 1)
    returnimage *= 255
    # Replace white pixels
    returnimage[whitemask] = image[whitemask]
    return returnimage.astype(np.uint8)
 