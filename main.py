import joblib
from PIL import Image, ImageOps
import numpy as np
import os
from skimage import feature
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
from itertools import groupby


class getGrayRumatrix:
    data = 0 
    def read_img(self,path=" "):
        
        try:
            img = Image.open(path) 
            img = img.convert('L')
            self.data=np.array(img)
            
        except:
            img = None
            
    def getGrayLevelRumatrix(self, array, theta):
            '''
            
            array: the numpy array of the image
            theta: Input, the angle used when calculating the gray scale run matrix, list type, can contain fields:['deg0', 'deg45', 'deg90', 'deg135']
            glrlm: output,the glrlm result
            '''
            P = array
            x, y = P.shape
            min_pixels = np.min(P)   # the min pixel
            run_length = max(x, y)   # Maximum parade length in pixels
            num_level = np.max(P) - np.min(P) + 1   # Image gray level
    
            deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]   # 0deg
            deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]   # 90deg
            diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0]+1, P.shape[1])]   #45deg
            deg45 = [n.tolist() for n in diags]
            Pt = np.rot90(P, 3)   # 135deg
            diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0]+1, Pt.shape[1])]
            deg135 = [n.tolist() for n in diags]
    
            def length(l):
                if hasattr(l, '__len__'):
                    return np.size(l)
                else:
                    i = 0
                    for _ in l:
                        i += 1
                    return i
    
            glrlm = np.zeros((num_level, run_length, len(theta)))   
            for angle in theta:
                for splitvec in range(0, len(eval(angle))):
                    flattened = eval(angle)[splitvec]
                    answer = []
                    for key, iter in groupby(flattened):  
                        answer.append((key, length(iter)))   
                    for ansIndex in range(0, len(answer)):
                        glrlm[int(answer[ansIndex][0]-min_pixels), int(answer[ansIndex][1]-1), theta.index(angle)] += 1   
            return glrlm
            
    def apply_over_degree(self,function, x1, x2):
        rows, cols, nums = x1.shape
        result = np.ndarray((rows, cols, nums))
        for i in range(nums):
                #print(x1[:, :, i])
                result[:, :, i] = function(x1[:, :, i], x2)
               # print(result[:, :, i])
                result[result == np.inf] = 0
                result[np.isnan(result)] = 0
        return result 
    def calcuteIJ (self,rlmatrix):
        gray_level, run_length, _ = rlmatrix.shape
        I, J = np.ogrid[0:gray_level, 0:run_length]
        return I, J+1

    def calcuteS(self,rlmatrix):
        return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]

    #1.SRE
    def getShortRunEmphasis(self,rlmatrix):
            I, J = self.calcuteIJ(rlmatrix)
            numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
            S = self.calcuteS(rlmatrix)
            return numerator / S
    #2.LRE
    def getLongRunEmphasis(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    #3.GLN
    def getGrayLevelNonUniformity(self,rlmatrix):
        G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
        numerator = np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    # 4. RLN
    def getRunLengthNonUniformity(self,rlmatrix):
            R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
            numerator = np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0]
            S = self.calcuteS(rlmatrix)
            return numerator / S

        # 5. RP
    def getRunPercentage(self,rlmatrix):
            gray_level, run_length,_ = rlmatrix.shape
            num_voxels = gray_level * run_length
            return self.calcuteS(rlmatrix) / num_voxels

        # 6. LGLRE
    def getLowGrayLevelRunEmphasis(self,rlmatrix):
            I, J = self.calcuteIJ(rlmatrix)
            numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
            S = self.calcuteS(rlmatrix)
            return numerator / S

        # 7. HGL   
    def getHighGrayLevelRunEmphais(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

        # 8. SRLGLE
    def getShortRunLowGrayLevelEmphasis(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    # 9. SRHGLE
    def getShortRunHighGrayLevelEmphasis(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (I*I))
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
 
    # 10. LRLGLE
    def getLongRunLow(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (J*J))
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
 
    # 11. LRHGLE
    def getLongRunHighGrayLevelEmphais(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S


def result(prediction):
    if prediction[0] == 0:
        print("HASIL PREDIKSI: Corrosion")
    else:
        print("HASIL PREDIKSI: NoCorrosion")

def glrlm_prediction(image_path):
    glr_matrix_calculator = getGrayRumatrix()
    glr_matrix_calculator.read_img(image_path)

    target_size = (256, 256)
    image = Image.open(image_path)
    
    # Resize Image
    image_new = ImageOps.fit(image, target_size, method=0, bleed=0.0, centering=(0.5, 0.5))
    temp_image_path = './temp_resized_image.jpg'
    image_new.save(temp_image_path)

    model_filename = './model/mlp_glrlm_model.pkl'
    loaded_mlp_classifier = joblib.load(model_filename)

    theta = ['deg0', 'deg45', 'deg90', 'deg135']
    glrlm_result = glr_matrix_calculator.getGrayLevelRumatrix(glr_matrix_calculator.data, theta)

    sre_result = glr_matrix_calculator.getShortRunEmphasis(glrlm_result)
    lre_result = glr_matrix_calculator.getLongRunEmphasis(glrlm_result)
    gln_result = glr_matrix_calculator.getGrayLevelNonUniformity(glrlm_result)
    rln_result = glr_matrix_calculator.getRunLengthNonUniformity(glrlm_result)
    rp_result = glr_matrix_calculator.getRunPercentage(glrlm_result)
    lglre_result = glr_matrix_calculator.getLowGrayLevelRunEmphasis(glrlm_result)
    hgl_result = glr_matrix_calculator.getHighGrayLevelRunEmphais(glrlm_result)
    srlgle_result = glr_matrix_calculator.getShortRunLowGrayLevelEmphasis(glrlm_result)
    srhgle_result = glr_matrix_calculator.getShortRunHighGrayLevelEmphasis(glrlm_result)
    lrlgle_result = glr_matrix_calculator.getLongRunLow(glrlm_result)
    lrhgle_result = glr_matrix_calculator.getLongRunHighGrayLevelEmphais(glrlm_result)

    # features = [sre_result, lre_result, gln_result, rln_result, rp_result, lglre_result,
    #                                 hgl_result, srlgle_result, srhgle_result, lrlgle_result, lrhgle_result]

    print(sre_result)

    features =  {
                "SRE": sre_result,
                "LRE": lre_result,
                "GLN": gln_result,
                "RLN": rln_result,
                "RP": rp_result,
                "LGLRE": lglre_result,
                "HGL": hgl_result,
                "SRLGLE": srlgle_result,
                "SRHGLE": srhgle_result,
                "LRLGLE": lrlgle_result,
                "LRHGLE": lrhgle_result,
            }

    transformed_features = {}

    # Define a list of angles
    angles = ['0', '45', '90', '135']

    # Iterate through the features and angles to create new labels
    for feature_name, feature_values in features.items():
        for angle, value in zip(angles, feature_values):
            new_label = f'{feature_name}{angle}'
            transformed_features[new_label] = value

    # Print or use transformed_features as needed
    print(transformed_features)

    desired_features = [
        "SRLGLE90",
        "SRLGLE0",
        "SRLGLE45",
        "SRLGLE135",
        "SRE45",
        "SRE135",
        "RLN45",
        "LRE45",
        "RLN135",
        "LRE135",
        "GLN45",
        "RP45",
        "HGL135",
        "SRHGLE135",
        "LRE0"
    ]

    # Create a new dictionary with only the desired features
    selected_features = {key: value for key, value in transformed_features.items() if key in desired_features}

    # Transform the dictionary to a 1D NumPy array
    transformed_features_array = np.array(list(selected_features.values())).reshape(1, -1)


    # Make predictions using the loaded model
    predictions = loaded_mlp_classifier.predict(transformed_features_array)

    ## Print Out Result
    result(predictions)

    # Close and remove the temporary image file
    image_new.close()
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)


def glcm_prediction(image_path):
    # Load the trained MLP classifier model from a file
    model_filename = './model/mlp_classifier_model.joblib'
    loaded_mlp_classifier = joblib.load(model_filename)

    target_size = (256, 256)

    image = Image.open(image_path)

    # Resize Image
    image_new = ImageOps.fit(image, target_size, method=0, bleed=0.0, centering=(0.5, 0.5))
    temp_image_path = './temp_resized_image.jpg'
    image_new.save(temp_image_path)

    def compute_glcm_features(image_path):
        img = io.imread(image_path)
        gray = color.rgb2gray(img)
        image = img_as_ubyte(gray)
        
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_cooccurrence = graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)

        contrast = graycoprops(matrix_cooccurrence, 'contrast')
        dissimilarity = graycoprops(matrix_cooccurrence, 'dissimilarity')
        homogeneity = graycoprops(matrix_cooccurrence, 'homogeneity')
        energy = graycoprops(matrix_cooccurrence, 'energy')
        correlation = graycoprops(matrix_cooccurrence, 'correlation')
        asm = graycoprops(matrix_cooccurrence, 'ASM')

        return {
            "Contrast": contrast,
            "Dissimilarity": dissimilarity,
            "Homogeneity": homogeneity,
            "Energy": energy,
            "Correlation": correlation,
            "ASM": asm
        }


    # Extract GLCM Value
    features = compute_glcm_features(temp_image_path)

    # Initialize a new dictionary to store the transformed features
    transformed_features = {}

    # Define a list of angles
    angles = ['0', '45', '90', '135']

    # Iterate through the features and angles to create new labels
    for feature_name, feature_values in features.items():
        for i, angle in enumerate(angles):
            new_label = f'{feature_name}{angle}'
            transformed_features[new_label] = feature_values[0][i]

    filtered_data = {key: value for key, value in transformed_features.items() if key in ['Correlation135', 'Contrast135', 'Homogeneity135', 'Dissimilarity135', 'Energy135', 'Correlation90', 'Contrast90', 'Contrast0', 'ASM135', 'Energy0', 'ASM0', 'Energy90', 'Homogeneity0', 'Dissimilarity90', 'Correlation0']}

    # Transform the dictionary to a 1D NumPy array
    transformed_features_array = np.array(list(filtered_data.values())).reshape(1, -1)


    # Make predictions using the loaded model
    predictions = loaded_mlp_classifier.predict(transformed_features_array)

    ## Print Out Result
    result(predictions)

    # Close and remove the temporary image file
    image_new.close()
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)



def lbp_prediction(image_path):
    # Load and preprocess a single image
    im = Image.open(image_path).convert('L')
    data = np.array(im)

    def lbp_texture(image):
        # Perform LBP feature extraction on a single image
        lbp = feature.local_binary_pattern(image, n_point, radius, 'default')
        max_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=max_bins, density=True)
        return hist


    # Define LBP parameters
    radius = 2
    n_point = radius * 8

    # Call the LBP function on the single image
    lbp_features = lbp_texture(data)

    loaded_model = joblib.load('model/mlp_lbp_model.pkl')
    prediction = loaded_model.predict([lbp_features])

    result(prediction)



image_path = './resize_data/NOCORROSION/100e35cf19.jpg'

glcm_prediction(image_path)
lbp_prediction(image_path)
glrlm_prediction(image_path)