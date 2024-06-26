# Ray tracing
# Pedro Velarde, UPM, 2024
# Ejemplo con densidad fijada analíticamente.
# Para TFG de Marco Antonio Buitrago López
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as const
from matplotlib.colors import BoundaryNorm
import math


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


def rotate_coordinates(x, y, cx, cy, angle_degrees):
    # Convertir ángulo de grados a radianes
    angle_rad = math.radians(angle_degrees)

    # Calcular las coordenadas rotadas
    x_new = cx + (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad)
    y_new = cy + (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad)

    return x_new, y_new


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


def ndens(x, y, z, L, Rs, center):
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    ndens = min(1, np.exp(- 20 * (r - (Rs / 2)) / L))
    gxn = (- 20 * ndens / L) * ((x - center[0]) / r)
    gyn = (- 20 * ndens / L) * ((y - center[1]) / r)
    gzn = 0

    return ndens, gxn, gyn, gzn


def interpolacionndens(coordenadas, nelectronica):  # LA NELECTRONICA DEBE SER UNA MATRIZ NP.ARRAY
    x = coordenadas[0]  # Obtener el valor de x de las coordenadas
    y = coordenadas[1]  # Obtener el valor de y de las coordenadas

    # Extraer las columnas de la matriz nelectronica
    xs = [fila[0] for fila in nelectronica]
    ys = [fila[1] for fila in nelectronica]

    # Encontrar el índice del valor más cercano por debajo y por arriba de x
    indice_x_por_debajo = max([i for i, valor_x in enumerate(xs) if valor_x <= x])
    indice_x_por_arriba = min([i for i, valor_x in enumerate(xs) if valor_x >= x])

    # Encontrar el índice del valor más cercano por debajo y por arriba de y
    indice_y_por_debajo = max([i for i, valor_y in enumerate(ys) if valor_y <= y])
    indice_y_por_arriba = min([i for i, valor_y in enumerate(ys) if valor_y >= y])

    # Obtener los valores correspondientes de x y y por debajo y por arriba
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
            print("La coordenada {} no se encuentra en la matriz.".format(coord))

    # Cálculo de áreas
    area1 = (valcoordndens[1] - x) * (y - valcoordndens[2])
    area2 = (x - valcoordndens[0]) * (y - valcoordndens[2])
    area3 = (x - valcoordndens[0]) * (valcoordndens[3] - y)
    area4 = (valcoordndens[1] - x) * (valcoordndens[3] - y)
    areatotal = area1 + area2 + area3 + area4

    # Densidades del cuadrilátero
    n1 = nelectronica[indice[0], 2]
    n2 = nelectronica[indice[1], 2]
    n3 = nelectronica[indice[2], 2]
    n4 = nelectronica[indice[3], 2]

    naproximada = ((area1 / areatotal) * n1) + ((area2 / areatotal) * n2) + ((area3 / areatotal) * n3) + (
                (area4 / areatotal) * n4)

    return naproximada


# frecuencia de absorción e-i
def colfrec(xr, yr, zr, nden, L, w, nc):
    Z = 0.5
    T = 100000  # eV
    coulog = 1  # logaritmo de Coulomb
    A = 20000

    nu = 3.2e-9 * Z * Z * nden * nc * coulog / A / T ** 1.5
    return nu


# Define la función que representa el sistema de ODEs
def rayeq(t, f, l, L, w, nc):
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
    nden, gxn, gyn, gzn = ndens(xr, yr, zr, L, Rs, center)
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


def show(L):
    # Dibujar el mapa de colores de $N_e/N_c$
    NX = NY = 1000
    nd = np.zeros((NX, NY))
    X = np.linspace(limitesx[0], limitesx[1], NX)
    Y = np.linspace(limitesy[0], limitesy[1], NY)
    plt.figure(figsize=(12, 8))

    for i in range(NX):
        for j in range(NY):
            nd[j, i], gxn, gyn, gzn = ndens(X[i], Y[j], 0, L, Rs, center)

    plt.contourf(X, Y, nd, 10)
    colorbar = plt.colorbar()
    colorbar.set_label('$N_e / N_c$', fontsize=14)

    # Asignar color rojo a valores mayores o iguales a 1
    nd_masked = np.ma.masked_where(nd < 1, nd)
    plt.contourf(X, Y, nd_masked, colors='red')

    plt.xlabel('X (m)', fontsize=14)
    plt.ylabel('Y (m)', fontsize=14)
    plt.xlim(X[0], X[-1])
    plt.ylim(Y[0], Y[-1])
    plt.title('Trayectoria de rayos', fontsize=22, pad=20)


radiolente = 10  # Radio del lente
NR = 100  # número de rayos (número de rectas)
Rss = 4.e-3  # Radio de la sección de estudio
Rs = Rss * 2  # Diámetro de la sección
semiangulofoco = 8
alfa = np.radians(semiangulofoco)  # Semiángulo del foco SOLO SE PUEDE CAMBIAR ALFA PARA AUMENTAR O DISMINUIR ZONA
# ILUMINAADA
L = Rss / np.tan(alfa)  # Distancia del lente al foco
l = 1.e-6  # longitud de onda del láser [m]
T = 2 * const.pi * L / const.c  # Valor máximo de la variable s [s]
num_t = 10000  # Cantidad de valores de la variable s
t = np.linspace(0, T, num_t)  # variable s
w = 2 * const.pi * const.c / l  # frecuencia en [rad/s] ($\omega$)
nc = const.m_e * const.epsilon_0 * (w / const.e) ** 2  # densidad crítica
k0 = 2 * const.pi / l  # vector de onda inicial
center = [3 * L / 4, Rs / 2]  # SOLO SE CAMBIA LA DISTANCIA DEL BLANCO CON RESPECTO AL EJE X
limitesx = [0, 2 * center[0]]
limitesy = [-limitesx[1] / 2 + Rs / 2, limitesx[1] / 2 + Rs / 2]
angles = [90, -45, -150]
NRtot = NR * len(angles)

# Define las condiciones iniciales y el rango de tiempo
ti = (0, T)  # Rango de s en el que resolver las ecuaciones de rayo

show(L)

# Resuelve el sistema de ODEs
method = 'Radau'  # método numérico de solución de la ODE
method = 'RK45'
method = 'BDF'

rays = init_rays_laser(NR, L, Rss, k0, center[0], center[1], 0, angles)  # condiciones iniciales

for i in range(NRtot):
    f0 = rays[:, i]  # Condiciones iniciales
    sol = solve_ivp(rayeq, ti, f0,
                    dense_output=True,
                    t_eval=t,
                    method=method,
                    args=(l, L, w, nc),
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
plt.savefig('Trayectorias.png', dpi=2000)

plt.show()
