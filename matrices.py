# Importamos las librerias de Python necesarias
import numpy as np
from skimage.io import imshow, imread
import cv2
import os


def redimensionar_y_recortar_central(ruta_img: str, ruta_img_salida: str) -> None:
    try:
        imagen = cv2.imread(ruta_img)
        alto, ancho, _ = imagen.shape

        # Para redimensionar la imagen, se achica para que la dimension mas pequeña sea igual a mil (ancho/alto)
        # Luego, se recorta la otra dimension para que sea igual a 500 pixeles
        if ancho > alto:
            factor_escala = 500 / alto
            nuevo_ancho = int(ancho * factor_escala)
            nuevo_alto = 500
        else:
            factor_escala = 500 / ancho
            nuevo_alto = int(alto * factor_escala)
            nuevo_ancho = 500

        # Redimensionar la imagen
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)

        # Recortar la imagen centralmente para obtener 500x500
        alto_redim, ancho_redim, _ = imagen_redimensionada.shape
        # Se obtienen estos dos valores para que la imagen resultante sea el centro de la imagen y lo 
        x_inicial = (ancho_redim - 500) // 2
        y_inicial = (alto_redim - 500) // 2
        imagen_central = imagen_redimensionada[y_inicial:y_inicial+500, x_inicial:x_inicial+500]

        # Guardar la imagen recortada
        cv2.imwrite(ruta_img_salida, imagen_central)

        print(f"Imagen redimensionada y recortada con éxito. Nueva imagen guardada en {ruta_img_salida}")
    except Exception as e:
        print(f"Ha ocurrido un error: {str(e)}")


def mostrar_imagen(imagen_bgr, factor_ampliacion):
  imagen = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB) # se convierte la imagen de bgr a rgb
  alto, ancho, canales = imagen.shape
  nueva_ancho = ancho * factor_ampliacion
  nueva_alto = alto * factor_ampliacion
  imagen_redimensionada = cv2.resize(imagen, (nueva_ancho, nueva_alto), interpolation=cv2.INTER_NEAREST)
  cv2.imshow('Imagen Color', imagen_redimensionada)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def separar_canales_rgb(imagen_bgr):
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    canal_rojo = imagen_rgb[:, :, 0]
    canal_verde = imagen_rgb[:, :, 1]
    canal_azul = imagen_rgb[:, :, 2] 

    return canal_rojo, canal_verde, canal_azul
  
def calcular_traspuesta(matriz):
    alto, ancho, z = matriz.shape
    aux = [[[0] * z for _ in range(alto)] for _ in range(ancho)]
    for i in range(alto):
        for j in range(ancho):
            aux[j][i] = matriz[i][j]
    return np.array(aux)

def escala_grises(matriz):
    alto, ancho, z = matriz.shape
    aux = matriz.copy()
    for i in range(alto):
        for j in range(ancho):
            r, g, b = int(matriz[i][j][0]), int(matriz[i][j][1]), int(matriz[i][j][2])
            gris = (r + g + b) // 3
            aux[i][j] = [gris, gris, gris]
    return np.array(aux)

def inversa(matriz):
    inversa = []
    try:
        inversa = np.linalg.inv(matriz)
    except np.linalg.LinAlgError:
        return None
    return np.array(inversa)

def convertir_a_2d(matriz_3d):
    matriz_2d = matriz_3d[:, :, 0]  # Como los 3 canales son iguales, elegimos el rojo
    return matriz_2d

def convertir_a_3d(matriz_2d):
    # Usamos np.stack para duplicar el único canal en tres (R, G, B)
    matriz_3d = np.stack((matriz_2d, matriz_2d, matriz_2d), axis=-1)
    return matriz_3d

def ajustar_contraste(imagen, alpha):
    imagen_ajustada = alpha * imagen
    
    # Asegurar que los valores estén en el rango [0, 255]
    imagen_ajustada = np.clip(imagen_ajustada, 0, 255).astype(np.uint8)
    
    return imagen_ajustada


redimensionar_y_recortar_central("img/fiera.png", "img_procesadas/fiera.png")
imagen_1 = imread("img_procesadas/fiera.png")
redimensionar_y_recortar_central("img/zalayeta.jpeg", "img_procesadas/zalayeta.jpeg")
imagen_2 = imread("img_procesadas/zalayeta.jpeg")
redimensionar_y_recortar_central("img/valentin.png", "img_procesadas/valentin.png")
imagen_3 = imread("img_procesadas/valentin.png")

imagen_1_traspuesta = calcular_traspuesta(imagen_1)
mostrar_imagen(imagen_1, 1)
mostrar_imagen(imagen_1_traspuesta, 1)
imagen_1_grises = escala_grises(imagen_1)
mostrar_imagen(imagen_1_grises, 1)
print(inversa(convertir_a_2d(imagen_1_grises)))
alpha_1 = 1.5
imagen_contraste_aumentado = ajustar_contraste(imagen_1_grises, alpha_1)
mostrar_imagen(imagen_contraste_aumentado, 1)
  
# print(imagen_1_traspuesta, "\n", imagen_1.shape)
# Notamos que se invierte la imagen (espejo) y se gira 90 grados en sentido antihorario