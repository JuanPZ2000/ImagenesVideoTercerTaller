# Juan Pablo Zuluaga C
import cv2
import numpy as np
import os
import sys
import metodos as mt


if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    assert image is not None, "No hay ninguna imagen en  {}".format(path_file)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    N = 2
    lista_descomposicion = mt.descomposicion(image_gray, N)
    num = 1
    for a in (lista_descomposicion):
        for b in a:
            cv2.imshow("imagen" + str(num), b)
            num += 1
    imagen_final= mt.interpolacion(lista_descomposicion[-1][-1], 4)
    cv2.imshow("Imagen interpolada",imagen_final)
    cv2.waitKey(0)






