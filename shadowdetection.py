import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import histogram

def cart2pol(x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y,x))
    return r,theta

def pol2cart(r,theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x,y








if __name__ == '__main__':


    # load the base image
    img = cv2.imread('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Computer Vision/Final Project/Datasets/DESOBA_Dataset/ShadowImage/000000003770.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

    # load the image of the main object in the scene (instance mask)
    instance = cv2.imread('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Computer Vision/Final Project/Datasets/DESOBA_Dataset/InstanceMask/000000003770.png')
    instance = cv2.cvtColor(instance, cv2.COLOR_BGR2RGB)

    # load the shadow mask
    shadow = cv2.imread('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Computer Vision/Final Project/Datasets/DESOBA_Dataset/ShadowMask/000000003770.png')
    shadow = cv2.cvtColor(shadow, cv2.COLOR_BGR2RGB)

    fig,ax = plt.subplots(1,3)
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(instance)
    ax[1].axis('off')
    ax[2].imshow(shadow)
    ax[2].axis('off')
    plt.show()

    # convert shadow mask to greyscale
    shadow_mask = np.sum(shadow, axis=2)

    plt.imshow(shadow_mask,cmap='gray')
    plt.colorbar()
    plt.show()

    # get locations of mask pixels
    rows, x = np.where(shadow_mask == 255)
    y = shadow_mask.shape[0] - rows



    # fit a line to the shadow points
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(0,x.max())
    y_fit = slope * x_fit + intercept

    plt.scatter(x,y,s=0.01,label='shadow')
    plt.plot(x_fit,y_fit, label='fit',color='r')
    plt.legend()
    plt.ylim([0,shadow.shape[0]])
    plt.xlim([0,shadow.shape[1]])
    plt.show()

    # bias the data by the intercept
    y_bias = y - intercept
    y_fit_bias = y_fit - intercept

    plt.scatter(x,y_bias,s=0.01,label='shadow')
    plt.plot(x_fit,y_fit_bias, label='fit',color='r')
    plt.legend()
    plt.ylim([0,shadow.shape[0]])
    plt.xlim([0,shadow.shape[1]])
    plt.show()

    # convert fit line to polar coords (degrees)
    r,theta = cart2pol(x_fit,y_fit_bias)

    # make angles relative to the vertical
    theta_shadow = np.mean(90 - theta[1:-1])

    # find the length of the shadow
    r_shadow = r[x_fit>x.min()].max() - x.min()

    # find height of the object
    object_mask = np.sum(instance, axis=2)
    object_height = np.sum(object_mask,axis=1)
    object_height = object_height[object_height>0]
    object_height = object_height.shape[0]

    # find zenith angle (relative to normal)
    a = object_height
    c = r_shadow
    B = theta_shadow

    # law of cosines to find 3rd side
    b2 = a**2 + c**2 - (2 * a * c * np.cos(np.radians(B)))
    b = np.sqrt(b2)

    # law of sines to find zenith angle
    C = np.degrees(np.arcsin( (c * np.sin(np.radians(B))) / b ))

    print(C)



