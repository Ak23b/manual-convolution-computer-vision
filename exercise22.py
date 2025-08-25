import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load our image and convert it appropriately
img = Image.open("Images/ftwu49b3.png").convert("L")
img_arr = np.array(img)

# Visualize our image
plt.imshow(img,cmap='Grays')
plt.title("Images/ftwu49b3.png")
plt.axis("off")
plt.show()

# Creating the kernel
kernel = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]])


# Creating the outputing to store outcome of the convolution
output = np.zeros_like(img_arr)


# Padding
pad = 1 # for the 3by3
img_padded = np.pad(img_arr,pad,mode="constant")

# Looping for the convolution
for i in range(img_arr.shape[0]):
    for j in range(img_arr.shape[1]):
        region = img_padded[i:i+3,j:j+3]
        output[i,j] = np.sum(region * kernel)

# Clipping the range
output = np.clip(0,255,output)

# Plot the convolved Image
plt.imshow(output)
plt.title("Convolved Image")
plt.axis("off")
plt.show()
