# Ray tracing
# Pedro Velarde, UPM, 2024
# Ejemplo con densidad fijada analíticamente.
# Para TFG de Marco Antonio Buitrago López
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import scipy.constants as const
from matplotlib.colors import BoundaryNorm
import math

# Registra el tiempo de inicio
start_time = time.time()

def simpson_integration(a, b, particiones, f):
    h = (b - a) / (particiones - 1)

    x_values = np.linspace(a, b, particiones)
    y_values = f(x_values)

    integral = h / 3 * (y_values[0] + 4 * np.sum(y_values[1:-1:2]) + 2 * np.sum(y_values[2:-2:2]) + y_values[-1])

    return integral


def rayos_mono_energeticos(numerorayos, radioseccion, radiolente, semiangulofoco):
    liminf = 0  # Límite inferior de integración
    Rs = radioseccion  # Radio de la sección de estudio
    alfa = np.radians(semiangulofoco)  # Semiángulo del foco
    sigma = 2
    valueexp = 2  # Valor del exponente
    Io = 1.e+5  # Valor de la intensidad cuando r = 0
    coef = Io * 2 * np.pi
    trozos = 1000  # Particiones en el método de Simpson
    err = 1.e-7  # Margen de error

    Fb = Rs / np.tan(alfa)  # Distancia de la sección de estudio al foco
    Fl = radiolente / np.tan(alfa)  # Distancia del lente al foco
    H = Fl - Fb  # Distancia entre lente y la sección de estudio
    numray = numerorayos * 2  # Cantidad total de "r"

    func = lambda r: coef * r * (np.exp(-((r / sigma) ** valueexp)))  # Función a integrar

    totalenergy = simpson_integration(liminf, radioseccion, trozos, func)  # Energía de la sección de estudio

    rayenergy = totalenergy / numerorayos  # Energía de cada rayo

    # Número de radios desde a hasta b
    rad = np.zeros(numerorayos + 1)  # seminumray columnas

    func_g = np.zeros(numerorayos)  # Vector diferencia de energías

    for i in range(numerorayos):
        radprueba = rad[i] + radioseccion / 20
        func_g[i] = np.abs(simpson_integration(rad[i], radprueba, trozos, func) - rayenergy)
        while func_g[i] >= err:
            radprueba = radprueba - (simpson_integration(rad[i], radprueba, trozos, func) - rayenergy) / func(radprueba)
            func_g[i] = np.abs(simpson_integration(rad[i], radprueba, trozos, func) - rayenergy)

        if radprueba <= radioseccion:
            rad[i + 1] = radprueba

    # --------------------- Posicionamiento de los rayos ---------------------------#
    hrays = np.zeros(numray)  # Distancias entre cada r_j
    posicionvect = np.zeros(numray)  # Posición vertical de cada rayo
    j = 1

    # Rellenando las posiciones desde "a + 1" hasta "b"
    for i in range(numray // 2):
        hrays[i + numray // 2] = rad[i + 1] - rad[i]
        posicionvect[i + numray // 2] = rad[i] + hrays[i + numray // 2] / 2

    # Rellenando las posiciones desde "-b" hasta "a"
    for i in range(numray // 2):
        hrays[i] = hrays[i + numray - j]
        posicionvect[i] = - posicionvect[i + numray - j]
        j = j + 2

    # Matriz rayos: x, y, kx, ky
    matrixR = np.zeros((numray, 4))
    for i in range(0, numray):
        matrixR[i, 0] = H  # Posición X de cada rayo
        matrixR[i, 1] = - posicionvect[i]  # Posición Y de cada rayo
        matrixR[i, 2] = Fb  # Posición
        matrixR[i, 3] = posicionvect[i]

    # Se guarda fichero txt de los vectores
    np.savetxt('Matrixrays.txt', matrixR, header='X (u)\t\tY (u)\t\tKx (u)\t\tKy (u)'
               , delimiter='\t', fmt='%f', comments='')

    return matrixR

# Función para rotar una coordenada con respecto a un punto
def rotate_coordinates(x, y, cx, cy, angle_degrees):
    # Convertir ángulo de grados a radianes
    angle_rad = math.radians(angle_degrees)

    # Calcular las coordenadas rotadas
    x_new = cx + (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad)
    y_new = cy + (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad)

    return x_new, y_new

# Función que da como salida los datos iniciales del disparo de rayos
def init_rays_laser(NR, L, Rss, k0, cx, cy, cz, angle_degrees_list):

    # Cargar el archivo de texto, ignorando la primera fila
    data = np.transpose(rayos_mono_energeticos(int(NR / 2), Rss, radiolente, semiangulofoco))
    num_rotations = len(angle_degrees_list)  # Número de rotaciones
    newNR = NR * num_rotations  # Númer0 de columnas de la matriz rays

    rays = np.zeros((7, newNR))

    rays[0, :] = 0.0  # x0
    rays[2, :] = 0.0  # z0
    rays[6, :] = 1.0  # Intensidad I/I0 de partida
    rays[5, :] = 0.0  # kz0

    for r in range(num_rotations):
        angle_degrees = angle_degrees_list[r]
        # Calcular índices para esta rotación en rays
        start_idx = r * NR
        end_idx = (r + 1) * NR

        # Inicializar las coordenadas para esta rotación
        rays[1, start_idx:end_idx] = data[1, :] + Rss  # y0
        rays[3, start_idx:end_idx] = data[2, :]  # kx
        rays[4, start_idx:end_idx] = data[3, :]  # ky

        # Normalizar kx, ky
        norm = np.sqrt(rays[3, start_idx:end_idx] ** 2 + rays[4, start_idx:end_idx] ** 2)
        rays[3, start_idx:end_idx] = rays[3, start_idx:end_idx] / norm
        rays[4, start_idx:end_idx] = rays[4, start_idx:end_idx] / norm
        rays[3, start_idx:end_idx] = rays[3, start_idx:end_idx] * k0
        rays[4, start_idx:end_idx] = rays[4, start_idx:end_idx] * k0

        # Rotar las coordenadas (x0, y0) para esta rotación
        for i in range(NR):
            x0 = rays[0, start_idx + i]
            y0 = rays[1, start_idx + i]

            # Rotar las coordenadas (x0, y0)
            x_n, y_n = rotate_coordinates(x0, y0, cx, cy, angle_degrees)

            # Actualizar las coordenadas rotadas en rays
            rays[0, start_idx + i] = x_n
            rays[1, start_idx + i] = y_n

            # Rotar las componentes kx, ky para esta rotación
            kx = rays[3, start_idx + i]
            ky = rays[4, start_idx + i]

            # Rotar las coordenadas (kx0, ky0)
            kx_new, ky_new = rotate_coordinates(kx, ky, cx, cy, angle_degrees)

            # Actualizar las componentes rotadas en rays
            rays[3, start_idx + i] = kx_new
            rays[4, start_idx + i] = ky_new

    np.savetxt('Rayos_en_t0.txt', np.transpose(rays), fmt='%.5f')

    return rays

# Función de densidad interpolada a partir de un mallado de densidades electrónicas

def ninterpolada(x, y, z, malla_dens, dif_x):
    deltaxy = dif_x / 100
    naprox_0 = interpolacionndens((x, y), malla_dens)
    naprox_x = interpolacionndens((x + deltaxy, y), malla_dens)
    naprox_y = interpolacionndens((x, y + deltaxy), malla_dens)
    gxn = (naprox_x - naprox_0) / deltaxy
    gyn = (naprox_y - naprox_0) / deltaxy
    gzn = 0
    return naprox_0, gxn, gyn, gzn
def mallado(num_mall, limx, limy):
    xparticiones = np.linspace(limx[0], limx[1], num_mall + 1)
    yparticiones = np.linspace(limy[0], limy[1], num_mall + 1)
    area_celda = np.abs((xparticiones[1] - xparticiones[0]) * (yparticiones[1] - yparticiones[0]))
    matriz_densidad_media = np.zeros((num_mall, num_mall))
    for i in range(num_mall):
        for j in range(num_mall):
            result = dblquad(lambda x, y: min(1, np.exp(
                - 20 * (np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) - (radioblanco / 2)) / L)),
                    yparticiones[j], yparticiones[j + 1], xparticiones[i], xparticiones[i + 1])
            matriz_densidad_media[i, j] = result[0] / area_celda

    # Calcular los centros de las celdas
    x_centros = (xparticiones[:-1] + xparticiones[1:]) / 2
    y_centros = (yparticiones[:-1] + yparticiones[1:]) / 2

    # Crear la matriz resultado (num_mall * num_mall, 3)
    resultado = np.zeros((num_mall * num_mall, 3))

    # Llenar la matriz resultado con los valores X, Y y N
    indice = 0
    for i in range(num_mall):
        for j in range(num_mall):
            resultado[indice, 0] = x_centros[j]
            resultado[indice, 1] = y_centros[i]
            resultado[indice, 2] = matriz_densidad_media[i, j]
            indice += 1

    resultado = np.array(resultado)
    # Guardar la matriz en un archivo de texto con 4 decimales
    np.savetxt('matriz_densidad_centros.txt', resultado, fmt='%.4f', header='X Y N')

    # Discretizar los colores en diez niveles
    boundaries = np.linspace(0, 1, 11)
    cmap = plt.cm.viridis
    norm = BoundaryNorm(boundaries, cmap.N)

    # Calcular los límites exactos de los ejes X e Y
    xmin, xmax = limx
    ymin, ymax = limy

    # Crear el gráfico de la malla con colores asignados
    plt.figure(figsize=(8, 6))
    plt.imshow(matriz_densidad_media, cmap=cmap, norm=norm, interpolation='nearest', origin='lower', extent=[xmin, xmax, ymin, ymax])

    # Agregar barra de color para indicar los valores
    colorbar = plt.colorbar()
    colorbar.set_label('$N_e / N_c$', fontsize=14)

    # Añadir título y etiquetas de los ejes
    plt.title('Ray tracing', fontsize=22, pad=20)
    plt.xlabel('X (m)', fontsize=14)
    plt.ylabel('Y (m)', fontsize=14)

    # Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
    plt.savefig('Mallado densidad.png', dpi=2000)

    ancho_celda = xparticiones[1] - xparticiones[0]
    return resultado, ancho_celda

# Función de interpolación de densidad electrónica
def interpolacionndens(coordenadas, nelectronica):
    x = coordenadas[0]
    y = coordenadas[1]

    # Extraer las columnas de la matriz nelectronica
    xs = [fila[0] for fila in nelectronica]
    ys = [fila[1] for fila in nelectronica]

    # Verificar si las coordenadas están dentro del rango de la matriz
    if x < min(xs) or x > max(xs) or y < min(ys) or y > max(ys):
        return 0  # Asignar 0 si está fuera del rango

    try:
        # Encontrar el índice del valor más cercano por debajo y por arriba de x
        indice_x_por_debajo = max([i for i, valor_x in enumerate(xs) if valor_x <= x])
        indice_x_por_arriba = min([i for i, valor_x in enumerate(xs) if valor_x >= x])

        # Encontrar el índice del valor más cercano por debajo y por arriba de y
        indice_y_por_debajo = max([i for i, valor_y in enumerate(ys) if valor_y <= y])
        indice_y_por_arriba = min([i for i, valor_y in enumerate(ys) if valor_y >= y])
    except ValueError as e:
        print(f"Error: {e}. Asegúrate de que las coordenadas estén dentro del rango de los datos.")
        return 0

    # Obtener los valores correspondientes de x e y por arriba y abajo
    valcoordndens = [xs[indice_x_por_debajo], xs[indice_x_por_arriba], ys[indice_y_por_debajo], ys[indice_y_por_arriba]]

    # Coordenadas del cuadrilátero
    coord1 = [valcoordndens[0], valcoordndens[3]]
    coord2 = [valcoordndens[1], valcoordndens[3]]
    coord3 = [valcoordndens[1], valcoordndens[2]]
    coord4 = [valcoordndens[0], valcoordndens[2]]

    # Rutina para obtener el índice de la fila dada una coordenada para la matriz nelectronica
    def encontrar_indice_fila(matriz, x, y):
        for i, fila in enumerate(matriz):
            if fila[0] == x and fila[1] == y:
                return i
        return -1

    # Obtener índice de la fila donde se encuentra las coordenadas
    indices = []
    for coord in [coord1, coord2, coord3, coord4]:
        indice = encontrar_indice_fila(nelectronica, coord[0], coord[1])
        if indice != -1:
            indices.append(indice)
        else:
            print(f"La coordenada {coord} no se encuentra en la matriz.")
            return 0  # Asignar 0 si alguna coordenada no se encuentra

    if len(indices) < 4:
        print("No se encontraron todas las coordenadas necesarias.")
        return 0

    # Cálculo de áreas
    area1 = (valcoordndens[1] - x) * (y - valcoordndens[2])
    area2 = (x - valcoordndens[0]) * (y - valcoordndens[2])
    area3 = (x - valcoordndens[0]) * (valcoordndens[3] - y)
    area4 = (valcoordndens[1] - x) * (valcoordndens[3] - y)
    areatotal = area1 + area2 + area3 + area4

    # Densidades del cuadrilátero
    n1 = nelectronica[indices[0], 2]
    n2 = nelectronica[indices[1], 2]
    n3 = nelectronica[indices[2], 2]
    n4 = nelectronica[indices[3], 2]

    naproximada = ((area1 / areatotal) * n1) + ((area2 / areatotal) * n2) + ((area3 / areatotal) * n3) + ((area4 / areatotal) * n4)

    return naproximada


# Función de la variable de absorción de energía. Frecuencia de absorción e-i.
def colfrec(xr, yr, zr, nden, L, w, nc):
    Z = 10
    T = 1000  # eV
    coulog = 10  # logaritmo de Coulomb
    A = 20

    nu = 3.2e-9 * Z * Z * nden * nc * coulog / A / T ** 1.5
    return nu


# Define la función que representa el sistema de ODEs
def rayeq(t, f, l, L, w, nc, matriz_densidad_media, ancho_celda):
    """
    Solución de:
    \begin{equation}
        \frac{d\vec r}{ds}=\frac{c^2}{\omega}\vec k \\
        \frac{d\vec k}{ds}=-\frac{\omega}{2}\nabla\frac{N_e}{N_c}
    \end{equation}
    con $N_c=\frac{m_e\epsilon_0\omega^2}{e^2}$
    """
    a = const.c ** 2 / w  # $c^2/\omega$
    b = 0.5 * w  # $\omega/2$

    xr, yr, zr, kx, ky, kz, I = f[0], f[1], f[2], f[3], f[4], f[5], f[6]
    mat_dens_media = matriz_densidad_media
    anch_cel = ancho_celda
    nden, gxn, gyn, gzn = ninterpolada(xr, yr, zr, mat_dens_media, anch_cel)
    nu = colfrec(xr, yr, zr, nden, L, w, nc)

    # Cálculo de derivadas
    dxdt = a * kx
    dydt = a * ky
    dzdt = a * kz
    dkxdt = -b * gxn
    dkydt = -b * gyn
    dkzdt = -b * gzn
    dIdt = -nu * nden * I
    # Habría que chequear que se verifica $|\vec{k}|=\omega/c\sqrt{1-N_e/N_c}$
    # que es una restricción al módulo del vector k
    return [dxdt, dydt, dzdt, dkxdt, dkydt, dkzdt, dIdt]


radiolente = 10  # Radio del lente [m]
NR = 50  # número de rayos (número de curvas)
Rss = 18.e-3  # Radio de la sección de estudio [m]

Rs = Rss * 2  # Diámetro de la sección [m]
semiangulofoco = 8  # [degree]
alfa = np.radians(semiangulofoco)  # Semiángulo del foco SOLO SE PUEDE CAMBIAR ALFA PARA AUMENTAR O DISMINUIR ZONA
# ILUMINAADA
L = Rss / np.tan(alfa)  # Distancia del lente al foco [m]
l = 1.e-6  # longitud de onda del láser [m]
radioblanco = 54.e-3  # [m]
T = 2 * const.pi * L / const.c  # Valor máximo de la variable s [s]
num_t = 2000  # Cantidad de valores de la variable s
t = np.linspace(0, T, num_t)  # variable s
w = 2 * const.pi * const.c / l  # frecuencia en [rad/s] ($\omega$)
nc = const.m_e * const.epsilon_0 * (w / const.e) ** 2  # densidad crítica
k0 = 2 * const.pi / l  # vector de onda inicial
center = [1.5 * radioblanco, Rs / 2]  # SOLO SE CAMBIA LA DISTANCIA DEL BLANCO CON RESPECTO AL EJE X
limitesx = [0, 2 * center[0]]
limitesy = [-limitesx[1] / 2 + Rs / 2, limitesx[1] / 2 + Rs / 2]
angles = [45]
NRtot = NR * len(angles)

# Define las condiciones iniciales y el rango de tiempo
ti = (0, T)  # Rango de s en el que resolver las ecuaciones de rayo

numerodeparticiones = 50  # número de celdas totales = numeropart x numeropart

matriz_densidad_media, ancho_celda = mallado(numerodeparticiones, limitesx, limitesy)
np.savetxt('matriz_densidad_por_celda.txt', matriz_densidad_media, fmt='%.4f')

# Resuelve el sistema de ODEs
method = 'Radau'  # método numérico de solución de la ODE
#method = 'RK45'
#method = 'BDF'

rays = init_rays_laser(NR, L, Rss, k0, center[0], center[1], 0, angles)  # condiciones iniciales

# Resolución de cada rayo
for i in range(NRtot):
    print(i)
    f0 = rays[:, i]  # Condiciones iniciales
    sol = solve_ivp(rayeq, ti, f0,
                    dense_output=True,
                    t_eval=t,
                    method=method,
                    args=(l, L, w, nc, matriz_densidad_media, ancho_celda),
                    atol=1e-10,  # Tolerancia absoluta
                    rtol=1e-10  # Tolerancia relativa
                    )

    data = np.column_stack((sol.y[0], sol.y[1], sol.y[6]))  # matriz num_t x 3 del rayo i

    indices_a_eliminar = []

    # Iterar sobre cada fila de la matriz
    for i in range(data.shape[0]):
        # Verificar si la energía en la fila i es menor que 0
        if not (limitesy[0] <= data[i, 1] <= limitesy[1]) or not (limitesx[0] <= data[i, 0] <= limitesx[1]):
            indices_a_eliminar.append(i)

    data = np.delete(data, indices_a_eliminar, axis=0)

    # Extraer las columnas X, Y, y Z (energía)
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]

    # Definir los límites de los niveles de color
    levels = np.linspace(0, 1, 11)

    # Crear una normalización discreta para asignar colores a cada nivel
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap('inferno').N)

    # Normalizar los datos de Z para que estén en el rango [0, 1]
    Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())

    # Graficar los puntos con un gradiente de color según la energía
    plt.scatter(X, Y, c=Z_normalized, cmap='inferno', norm=norm, s=0.002)

plt.colorbar(label='Energía $E_t/E_{t0}$')

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Trayectoria de rayos.png', dpi=2000)

plt.show()

# Registra el tiempo de finalización
end_time = time.time()

# Calcula el tiempo transcurrido
elapsed_time = end_time - start_time
elapsed_time_rounded = round(elapsed_time, 3)
print(f"Tiempo de ejecución: {elapsed_time_rounded} segundos")