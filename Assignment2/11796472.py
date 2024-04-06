# Assignment 2: Fourier Transform & Filtering in Frequency Domain

# Name: Lucas Keiti Anbo Mihara 
# NUSP: 11796472 
# Course: SCC0251 
# Semester: 2024/1

import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from numpy import fft


def low_pass_filter(n, m, r):
    filter = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            filter[i][j] = 1 if np.sqrt((i - n/2.0)**2 + (j - m/2.0)**2) <= r else 0
    return filter


def high_pass_filter(n, m, r):
    filter = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            filter[i][j] = 1 if np.sqrt((i - n/2)**2 + (j - m/2)**2) > r else 0
    return filter


def band_stop_filter(n, m, r0, r1):
    filter = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            d = np.sqrt((i - n/2)**2 + (j - m/2)**2)
            filter[i][j] = 0 if d >= r1 and d <= r0  else 1
    return filter


def laplacian_filter(n, m):
    filter = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            filter[i][j] = -4 * (np.pi**2) * ((i - n/2)**2 + (j - m/2)**2)
    return filter


def gaussian_filter(n, m, sigma1, sigma2):
    filter = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            filter[i][j] = np.e ** -(((i - n/2)**2 / ((2 * sigma1**2)) + (j - m/2)**2 / (2 * sigma2**2)))
    return filter


def read_image(image_name):
    return imageio.imread(f"{image_name}")


def print_image(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap="gray")
    plt.axis('off') # remove axis with numbers


def root_mean_squared_error(result_image, reference_image):
    n = result_image.shape[0]
    sum = 0
    for i in range(n):
        for j in range(n):
            sum += (result_image[i][j] - reference_image[i][j]) ** 2
    return np.round((np.sqrt(sum))/n, 4)


def get_filter_from_input(image_n, image_m):
    index = int(input().rstrip())
    match index:
        case 0: # low-pass filter
            r = int(input().rstrip())
            return low_pass_filter(image_n, image_m, r)
        case 1: # high-pass
            r = int(input().rstrip())
            return high_pass_filter(image_n, image_m, r)
        case 2: # band-stop
            r0 = int(input().rstrip())
            r1 = int(input().rstrip())
            return band_stop_filter(image_n, image_m, r0, r1)
        case 3: # laplacian
            return laplacian_filter(image_n, image_m)
        case 4: # gaussian
            s0 = int(input().rstrip())
            s1 = int(input().rstrip())
            return gaussian_filter(image_n, image_m, s0, s1)


def filter_fft_image(image, filter):
    fourier_domain = fft.fft2(image)
    fourier_filtered = fft.fftshift(fourier_domain) * filter
    # fourier_filtered = fourier_domain * filter
    filtered_image = fft.ifft2(fft.ifftshift(fourier_filtered))
    # filtered_image = fft.ifft2(fourier_filtered)
    real_image = np.real(filtered_image)
    result = (real_image - np.min(real_image)) * (255 / (np.max(real_image) - np.min(real_image)))
    return result


def main():
    base_image_name = input().rstrip()
    reference_image_name = input().rstrip()

    base_image = read_image(base_image_name)
    reference_image = read_image(reference_image_name)

    filter = get_filter_from_input(base_image.shape[0], base_image.shape[1])

    result_image = filter_fft_image(base_image, filter)

    print(root_mean_squared_error(result_image, reference_image))


if __name__ == '__main__':
    main()

