import tkinter as tk # tkinter for python 3.x
from tkinter import filedialog, Menu # tkinter for python 3.
from PIL import Image, ImageTk # Pillow is a fork of the Python Imaging Library (PIL)
import cv2 # OpenCV
from cv2.ximgproc import guidedFilter 
import numpy as np
import matplotlib.pyplot as plt

import model

class HazeDensityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Haze Density Estimation App")

        # Title label
        self.title_label = tk.Label(root, text="Haze Density Estimation App", font=("Helvetica", 16, "bold"))
        self.title_label.pack()

        # Menu bar
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.browse_image)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        root.config(menu=menubar)

        self.image_path = None
        self.results_path = None

        self.label = tk.Label(root, text="Select Image:")
        self.label.pack()

        self.select_button = tk.Button(root, text="Browse", command=self.browse_image)
        self.select_button.pack()

        self.analyze_button = tk.Button(root, text="Analyze", command=self.analyze_image, state=tk.DISABLED)
        self.analyze_button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def browse_image(self):
        self.image_path = filedialog.askopenfilename(title="Select Image", filetypes=(("Image files", "*.png;*.jpg;*.bmp"), ("All files", "*.*")))
        if self.image_path:
            self.analyze_button.config(state=tk.NORMAL)

    def analyze_image(self):
        HazDesNet = model.load_HazDesNet()
        img = cv2.imread(self.image_path)

        guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, axis=0)
        
        haz_des_map = HazDesNet.predict(img)
        haz_des_map = haz_des_map[0,:,:,0]

        guide = cv2.resize(guide, (haz_des_map.shape[1], haz_des_map.shape[0]))
        
        haz_des_map = guidedFilter(guide=guide, src=haz_des_map, radius=32, eps=500)

        des_score = np.mean(haz_des_map)

        self.results_path = './results/result.png'
        cv2.imwrite(self.results_path, haz_des_map*255)

        self.result_label.config(text="The haze density score is: %.2f" % (des_score))

        plt.imshow(haz_des_map, cmap='jet')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = HazeDensityApp(root)
    root.mainloop()
