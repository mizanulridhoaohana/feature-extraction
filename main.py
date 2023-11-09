import joblib
from PIL import Image, ImageOps
import numpy as np
import os
from skimage import feature
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte


def result(prediction):
    if prediction[0] == 0:
        print("HASIL PREDIKSI: Corrosion")
    else:
        print("HASIL PREDIKSI: NoCorrosion")

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

    # Transform the dictionary to a 1D NumPy array
    transformed_features_array = np.array(list(transformed_features.values())).reshape(1, -1)

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



image_path = './data/NOCORROSION/100e35cf19.jpg'

glcm_prediction(image_path)
lbp_prediction(image_path)