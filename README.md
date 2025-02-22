HazDesNet - Haze Density Estimation

HazDesNet is a deep learning model designed for estimating haze density in images. It uses a convolutional neural network (CNN) to predict haze levels and visualize the results. This project includes a trained model and a simple GUI for user-friendly analysis.

Features

Uses Keras for deep learning model implementation.

Processes images to estimate haze density.

GUI built with Tkinter for easy image selection and analysis.

Visualization of haze density using Matplotlib.

Supports multiple image formats (.png, .jpg, .bmp).

Project Structure


 main.py        # Main script to run the model
 
 model.py       # Defines the HazDesNet model architecture
 
 requirements.txt # List of required Python packages
 
 ui.py          # GUI implementation using Tkinter
 
 weights.hdf5   # Pre-trained model weights

After running the application and selecting an image, the output will display the haze density score and a heatmap visualization.
