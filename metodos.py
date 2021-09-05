# Juan Pablo Zuluaga C, Sergio Hernandez
import cv2
import numpy as np
import os
import sys
def diezmado(image_gray,D): # image in BW , D>1
    assert (D >= 1 and type(D) is int), "El D debe ser un numero entero mayor a 1"
    # FFT
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

    # Pre- computations
    num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    half_size_y = num_rows / 2  # La imagen no es de las mismas dimensiones
    half_size_x = num_cols / 2

    # Filtering mask
    low_pass_mask = np.zeros_like(image_gray)
    freq_cut_off = 1/D  # it should less than 1
    radius_cut_off_x = int(freq_cut_off * half_size_x)
    radius_cut_off_y = int(freq_cut_off * half_size_y)
    x = ((col_iter - half_size_x) ** 2) / radius_cut_off_x ** 2
    y = ((row_iter - half_size_y) ** 2) / radius_cut_off_y ** 2
    idx_hp = (x + y) < 1
    low_pass_mask[idx_hp] = 1

    # Filtering
    mask = low_pass_mask
    fft_filtered = image_gray_fft_shift * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    #Decimation
    image_decimated = image_filtered[::D, ::D]
    return image_decimated

def interpolacion(image_gray, I):
    # Interpolacion
    assert (I >= 1 and type(I) is int), "El D debe ser un numero entero mayor a 1"
    rows, cols = image_gray.shape
    num_of_zeros = I
    image_zeros = np.zeros((num_of_zeros * rows, num_of_zeros * cols), dtype=image_gray.dtype)
    image_zeros[::num_of_zeros, ::num_of_zeros] = image_gray
    # FFT
    image_gray_fft = np.fft.fft2(image_zeros)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

    # Pre- computations
    num_rows, num_cols = (image_zeros.shape[0], image_zeros.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    half_size_y = num_rows / 2  # La imagen no es de las mismas dimensiones
    half_size_x = num_cols / 2

    # Filtering mask
    low_pass_mask = np.zeros_like(image_zeros)
    freq_cut_off = 1 / I  # it should less than 1
    radius_cut_off_x = int(freq_cut_off * half_size_x)
    radius_cut_off_y = int(freq_cut_off * half_size_y)
    x = ((col_iter - half_size_x) ** 2) / radius_cut_off_x ** 2
    y = ((row_iter - half_size_y) ** 2) / radius_cut_off_y ** 2
    idx_hp = (x + y) < 1
    low_pass_mask[idx_hp] = 1

    # Filtering
    mask = low_pass_mask
    fft_filtered = image_gray_fft_shift * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    return image_filtered

def descomposicion(image_gray, N):
    H = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    D = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
    L = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])

    descomposicion_final= []
    descomposicion_aux = []
    if N == 1:
        descomposicion_aux.append(diezmado(cv2.filter2D(image_gray, -1, H), 2))
        descomposicion_aux.append(diezmado(cv2.filter2D(image_gray, -1, V), 2))
        descomposicion_aux.append(diezmado(cv2.filter2D(image_gray, -1, D), 2))
        descomposicion_aux.append(diezmado(cv2.filter2D(image_gray, -1, L), 2))
        descomposicion_final.append(descomposicion_aux)
    else:
        for i in range(N):
            #Creo la lista para agregarla a la lista final
            descomposicion_aux = []
            # Convolucion y diezmado
            if i == 0:
                descomposicion_aux.append(diezmado(cv2.filter2D(image_gray, -1, H), 2))
                descomposicion_aux.append(diezmado(cv2.filter2D(image_gray, -1, V), 2))
                descomposicion_aux.append(diezmado(cv2.filter2D(image_gray, -1, D), 2))
                IL = diezmado(cv2.filter2D(image_gray, -1, L), 2)

            elif i == N-1:
                descomposicion_aux.append(diezmado(cv2.filter2D(IL, -1, H), 2))
                descomposicion_aux.append(diezmado(cv2.filter2D(IL, -1, V), 2))
                descomposicion_aux.append(diezmado(cv2.filter2D(IL, -1, D), 2))
                descomposicion_aux.append(diezmado(cv2.filter2D(IL, -1, L), 2))
            else:
                descomposicion_aux.append(diezmado(cv2.filter2D(IL, -1, H), 2))
                descomposicion_aux.append(diezmado(cv2.filter2D(IL, -1, V), 2))
                descomposicion_aux.append(diezmado(cv2.filter2D(IL, -1, D), 2))
                IL = diezmado(cv2.filter2D(IL, -1, L), 2)

            descomposicion_final.append(descomposicion_aux)
    return descomposicion_final