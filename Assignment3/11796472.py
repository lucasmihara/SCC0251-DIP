# Assignment 2: Color & Segmentation & Morphology

# Name: Lucas Keiti Anbo Mihara 
# NUSP: 11796472 
# Course: SCC0251 
# Semester: 2024/1

import numpy as np
import imageio.v3 as imageio
from skimage import morphology

def to_gray_scale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) <= 2:
        return image #Se for RGB converte para Gray, senão apenas mantém
    return np.dot(image, [0.2989, 0.5870, 0.1140]).astype(np.int64)


def filter_gaussian(P, Q):
    s1 = P
    s2 = Q

    D = np.zeros([P, Q])  # Compute Distances
    for u in range(P):
        for v in range(Q):
            x = (u-(P/2))**2/(2*s1**2) + (v-(Q/2))**2/(2*s2**2)
            D[u, v] = np.exp(-x)
    return D

def map_value_to_color(value, min_val, max_val, colormap):
    # Scale the value to the range [0, len(colormap) - 1]
    scaled_value = (value - min_val) / (max_val - min_val) * (len(colormap) - 1)
    # Determine the two closest colors in the colormap
    idx1 = int(scaled_value)
    idx2 = min(idx1 + 1, len(colormap) - 1)
    # Interpolate between the two colors based on the fractional part
    frac = scaled_value - idx1
    color = [
        (1 - frac) * colormap[idx1][0] + frac * colormap[idx2][0],
        (1 - frac) * colormap[idx1][1] + frac * colormap[idx2][1],
        (1 - frac) * colormap[idx1][2] + frac * colormap[idx2][2]
    ]
    return color

def create_heatmap(N, M):
    heatmap_colors = [
        [1, 0, 1],   # Pink
        [0, 0, 1],   # Blue
        [0, 1, 0],   # Green
        [1, 1, 0],   # Yellow
        [1, 0, 0]    # Red
    ]

    color_distribution = filter_gaussian(M, N)
    min_val = np.min(np.array(color_distribution))
    max_val = np.max(np.array(color_distribution))
    heatmap_image = np.zeros([M, N, 3]) #Imagem RGB vazia
    for i in range(M):
        for j in range(N):
            heatmap_image[i, j] = map_value_to_color(color_distribution[i, j], min_val, max_val, heatmap_colors)
    return heatmap_image

def erosion(image):
    structure = morphology.square(3)
    return morphology.binary_erosion(image, structure) * 255

def dilation(image):
    structure = morphology.square(3)
    return morphology.binary_dilation(image, structure) * 255

def thresholding(f, L):
    # create a new image with zeros
    f_tr = np.ones(f.shape).astype(np.uint8)
    # setting to 0 the pixels below the threshold
    f_tr[np.where(f < L)] = 0
    return f_tr

def otsu_thresholding(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2, dtype=np.uint8)
    else:
        gray = np.copy(image)

    # Calculate histogram
    hist, _ = np.histogram(gray, bins=np.arange(256))

    # Normalize histogram
    hist_norm = hist.astype(np.float32) / np.sum(hist)

    # Initialization
    threshold = 0
    max_var = 0

    # Compute inter-class variance for all possible thresholds
    for t in range(1, 255):
        # Background probabilities and means
        w0 = np.sum(hist_norm[:t])
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        u0 = np.sum(np.arange(t) * hist_norm[:t]) / w0
        u1 = np.sum(np.arange(t, 255) * hist_norm[t:]) / w1

        # Compute inter-class variance
        var = w0 * w1 * (u0 - u1) ** 2

        # Update threshold if variance is larger
        if var > max_var:
            max_var = var
            threshold = t

    # Apply thresholding
    thresholded = (gray > threshold) * 255

    return thresholded.astype(np.uint8)

def read_image(image_name):
    return imageio.imread(f"{image_name}")

def rms_error(img, out):
    M,N = img.shape
    error = ((1/(M*N))*np.sum((img-out)**2))**(1/2)
    return error

def color_rms_error(img, out):
    return np.round(np.average([rms_error(img[:,:,0], out[:,:,0]), rms_error(img[:,:,1], out[:,:,1]), rms_error(img[:,:,2], out[:,:,1])]), 4)

def get_images():
    base_image_name = input().rstrip()
    reference_image_name = input().rstrip()

    base_image = read_image(base_image_name)
    reference_image = read_image(reference_image_name)

    return base_image, reference_image

def get_indexes():
    indexes_input = input().rstrip()
    return indexes_input.split(' ')

def apply_commands(image, indexes):
    for index in indexes:
        match index:
            case '1':
                image = erosion(image)
            case '2':
                image = dilation(image)
    return image

def to_rgb(image):
    h, w = image.shape
    rgb_image = np.empty((h, w, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = image
    rgb_image[:, :, 1] = image
    rgb_image[:, :, 2] = image

    return rgb_image

def normalize(image):
    return (image - np.min(image)) * (255 / (np.max(image) - np.min(image)))

def apply_colored_mask(base_image, mask):
    heatmap = create_heatmap(base_image.shape[0], base_image.shape[1])
    result_image = np.zeros((base_image.shape[0], base_image.shape[1], 3))
    for i in range(base_image.shape[0]):
        for j in range(base_image.shape[1]):
            if mask[i][j] == 255:
                result_image[i][j][0] = 1
                result_image[i][j][1] = 1
                result_image[i][j][2] = 1
            else:
                result_image[i][j] = heatmap[i][j]
    return normalize(result_image).astype(np.uint8)

def combine_mask(base_image, colored_mask, alpha):
    return ((1 - alpha) * to_rgb(base_image) + alpha * colored_mask).astype(np.uint8)

def main():
    base_image, reference_image = get_images()
    indexes = get_indexes()

    base_image = to_gray_scale(base_image)
    mask = otsu_thresholding(base_image)
    mask = apply_commands(mask, indexes)
    colored_mask = apply_colored_mask(base_image, mask)
    final_image = combine_mask(base_image, colored_mask, 0.3)

    print(color_rms_error(final_image, reference_image))

if __name__ == '__main__':
    main()