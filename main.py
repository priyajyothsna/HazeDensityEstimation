import cv2 # OpenCV 
import numpy as np # Numpy
import scipy.io as scio # Scipy
import pandas as pd # Pandas
import matplotlib.pyplot as plt # Matplotlib
from cv2.ximgproc import guidedFilter # Guided Filter is a technique used to enhance the quality of the image
from keras.models import Model # Keras
from keras.layers import Conv2D, Input, Concatenate, MaxPool2D, BatchNormalization, AvgPool2D #convolutional operations

weight_path = 'weights.hdf5' # Path to the weights file

def HazDesNet(): # Function to create the HazDesNet model
    input_1 = Input(shape=(None, None, 3)) # Input layer of the model is a 3 channel image

    conv_1 = Conv2D(filters=24, kernel_size=(5, 5), strides=(1, 1))(input_1) # Convolutional layer with 24 filters and kernel size of 5x5

    conv_1 = Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1))(conv_1) # Convolutional layer with 24 filters and kernel size of 1x1

    max_pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1) # Max pooling layer with pool size of 2x2

    max_pool_1 = BatchNormalization()(max_pool_1) # Batch normalization layer

    conv_2 = Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1))(max_pool_1) # Convolutional layer with 48 filters and kernel size of 5x5

    max_pool_2 = MaxPool2D(pool_size=(5, 5), strides=(1, 1))(conv_2) # Max pooling layer with pool size of 5x5
    avg_pool_2 = AvgPool2D(pool_size=(5, 5), strides=(1, 1))(conv_2) # Average pooling layer with pool size of 5x5

    max_avg_pool = Concatenate()([max_pool_2, avg_pool_2]) # Concatenate the max pooling and average pooling layers

    conv_3 = Conv2D(filters=1, kernel_size=(6, 6), strides=(1, 1), activation='sigmoid')(max_avg_pool) # Convolutional layer with 1 filter and kernel size of 6x6

    model = Model(inputs=input_1, outputs=conv_3) # Create the model
    
    return model # Return the model

def load_HazDesNet(): # Function to load the HazDesNet model
    model = HazDesNet() # Create the model
    model.load_weights(weight_path) # Load the weights of the model
    return model # Return the model

if __name__ == "__main__": # Main function
    # Load model
    HazDesNet_model = load_HazDesNet() # Load the HazDesNet model
    HazDesNet_model.summary() # Print the summary of the model

    i = 1 # Image number
    image_path = './images/%d.bmp' % (i) # Path to the image is the images folder
    results_path = './results/%d.png' % (i) # Path to the results folder

    img = cv2.imread(image_path) # Read the image
    guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert the image to grayscale
    img = np.expand_dims(img, axis=0) # Expand the dimensions of the image
    
    # Predict one hazy image
    haz_des_map = HazDesNet_model.predict(img) # Predict the haze density map of the image
    haz_des_map = haz_des_map[0,:,:,0] # Get the haze density map of the image
    guide = cv2.resize(guide, (haz_des_map.shape[1], haz_des_map.shape[0])) # Resize the guide image to the size of the haze density map
    haz_des_map = guidedFilter(guide=guide, src=haz_des_map, radius=32, eps=500) # Apply guided filter to the haze density map

    des_score = np.mean(haz_des_map) # Calculate the haze density score of the image

    cv2.imwrite(results_path, haz_des_map*255) # Save the results to the results folder

    print("The haze density score of " + image_path + " is: %.2f" % (des_score)) # Print the haze density score of the image
    print("Save results to " + results_path) # Print the path to the results folder

    plt.imshow(haz_des_map, cmap='jet') # Plot the haze density map
    plt.show() # Show the plot


    y_pred = np.zeros((100, 1)) # Predicted scores
    data = scio.loadmat('./dataset/LIVE_Defogging/gt.mat') # Load the ground truth data
    y_true = data['subjective_study_mean'].T # Ground truth scores is the subjective study mean. subjective_study_mean is a 100x1 matrix that contains the ground truth scores of the 100 images

    for k in range(100): # Loop through the 100 images
        data_file = './dataset/LIVE_Defogging/%d.bmp' % (k + 1) # Path to the image
        img_test = cv2.imread(data_file) # Read the image
        img_test = np.expand_dims(img_test, axis=0) # Expand the dimensions of the image
        
        y_temp = HazDesNet_model.predict(img_test) # Predict the haze density map of the image
        y_pred[k, 0] = np.mean(y_temp) # Calculate the haze density score of the image

    df = pd.DataFrame({'true': y_true[:, 0], 'pred': y_pred[:, 0]}) # Create a dataframe with the ground truth and predicted scores
    print(df) # Print the dataframe
    #print("SROCC: %.4f" % df.corr('spearman').ix[[0]].values[0][1]) # Print the Spearman correlation coefficient
    #print("PLCC: %.4f" % df.corr('pearson').ix[[0]].values[0][1]) # Print the Pearson correlation coefficient
