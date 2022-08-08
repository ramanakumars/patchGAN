import numpy as np

'''Data Preprocessing'''
def crop_images_batch(inputs, target):
    batch_size = len(inputs)
    new_imgs   = np.zeros([batch_size, 4, 256, 256])
    new_masks = np.zeros([batch_size, 1, 256, 256])
    height, width = 350, 350
    xs, ys = np.random.uniform(low=0,high=int(height-256),size=batch_size), np.random.uniform(low=0,high=int(width-256),size=batch_size)
    for i in range(batch_size):
        x, y = xs[i], ys[i]
        start_x, end_x = int(x), int(x)+256
        new_imgs[i,:,:,:] = inputs[i,:,start_x : end_x, int(y): int(y)+256]/255.
        new_masks[i,:,:,:] =  target[i,:,int(x): int(x)+256,int(y): int(y)+256]

    return new_imgs, new_masks

