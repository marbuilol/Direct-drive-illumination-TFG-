# Ray tracing
# Pedro Velarde, UPM, 2024
# Ejemplo con densidad fijada analíticamente.
# Para TFG de Marco Antonio Buitrago López
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as const


def init_rays(NR, k0, L):
    rays = np.zeros((6, NR))
    i = 0.5 * np.pi * np.linspace(0, 1, NR)
    rays[0:2, :] = 0.0  # x0 y0 z0
    rays[3, :] = k0 * np.cos(i)  # kx0: rayos saliendo de (0,0) en abanico de 90º
    rays[4, :] = k0 * np.sin(i)  # ky0
    rays[5, :] = 0.0  # kz0
    return rays


def init_rays_parallel(NR, k0, L):
    rays = np.zeros((6, NR))
    i = 0.5 * np.pi * np.linspace(0, 1, NR)
    rays[0, :] = 0.0  # x0,y0,z0
    rays[1, :] = np.linspace(0, 1, NR) * L
    rays[2, :] = 0.0
    rays[3, :] = k0  # kx0: rayos saliendo de (0,0) en abanico de 90º
    rays[4, :] = 0  # ky0
    rays[5, :] = 0.0  # kz0
    return rays


def ndens1(xr, yr, zr, L):
    # esta función devuelve los gradientes de N_e/N_c
    # Idealmente debería ser una función de interpolación
    # de los valores de un array de NX*NY de N_e
    nx = 3
    ny = 5
    if 0 <= xr <= L and 0 <= yr <= L:
        ndensx = (1 + np.sin(nx * xr / L * np.pi))
        ndensy = (1 + np.sin(ny * yr / L * np.pi))
        ndens = min(1.0, xr / L * ndensx * ndensy)

        # cálculo de gradientes
        gxn = (ndensx + np.pi * nx * xr / L * np.cos(nx * xr / L * np.pi)) * ndensy / L
        gyn = xr / L * ndensx * ny * np.pi / L * np.cos(ny * yr / L * np.pi)
        gzn = 0
    else:  # aseguramos que los rayos no vuelvan si salen de la caja
        ndens = 0
        gxn = 0
        gyn = 0
        gzn = 0
    return ndens, gxn, gyn, gzn


def ndens(x, y, z, L):
    # esta función devuelve la densidad y sus gradientes análogos a density_profile

    centercoords = (L, L / 2)  # Centro del círculo
    center = [centercoords[0], centercoords[1]]
    max_density_radius = L / 2  # Puedes ajustar el radio máximo de densidad según tus necesidades
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    if 0 <= x <= L and 0 <= y <= L:
        ndens = min(1, (1.5 - (r / max_density_radius) ** 2))
        # Esta ndens es n/nc que estará comprendido entre 0 y 1
        # cálculo de gradientes
        gxn = (x - center[0]) / max_density_radius ** 2
        gyn = (y - center[1]) / max_density_radius ** 2
        gzn = 0
        if r > max_density_radius:
            ndens = 0
            gxn = 0
            gyn = 0
            gzn = 0

    else:  # aseguramos que los rayos no vuelvan si salen de la caja
        ndens = 0
        gxn = 0
        gyn = 0
        gzn = 0

    return ndens, gxn, gyn, gzn


# Define la función que representa el sistema de ODEs
def rayeq(t, f, l, L, w):
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

    xr, yr, zr, kx, ky, kz = f[0], f[1], f[2], f[3], f[4], f[5]
    nden, gxn, gyn, gzn = ndens(xr, yr, zr, L)
    # Cálculo de derivadas
    dxdt = a * kx
    dydt = a * ky
    dzdt = a * kz
    dkxdt = -b * gxn
    dkydt = -b * gyn
    dkzdt = -b * gzn
    # Habría que chequear que se verifica $|\vec{k}|=\omega/c\sqrt{1-N_e/N_c}$
    # que es una restricción al módulo del vector k
    return [dxdt, dydt, dzdt, dkxdt, dkydt, dkzdt]


def show(L):
    # Dibujar el mapa de colores de $N_e/N_c$
    NX = NY = 1000  # Precisión de dibujo
    nd = np.zeros((NX, NY))
    X = np.linspace(0.0, L, NX)
    Y = np.linspace(0.0, L, NY)
    for i in range(0, NX):
        for j in range(0, NY):
            nd[j, i], gxn, gyn, gzn = ndens(X[i], Y[j], 0, L)
    plt.contourf(X, Y, nd, 21)
    plt.colorbar(label='$N_e/N_c$')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(X[0], X[-1])
    plt.ylim(Y[0], Y[-1])
    plt.title('Trayectoria rayos y $N/Nc$')


NR = 100  # número de rayos
l = 1e-6  # longitud de onda del láser [m]
L = 200e-6  # tamaño del sistema en X o Y [m]
T = 2 * const.pi * L / const.c  # Valor máximo de la variable s [s]
t = np.linspace(0, T, 10000)  # variable s
w = 2 * const.pi * const.c / l  # frecuencia en [rad/s] ($\omega$)
nc = const.m_e * const.epsilon_0 * (w / const.e) ** 2  # densidad crítica
k0 = 2 * const.pi / l  # vector de onda inicial

# Define las condiciones iniciales y el rango de tiempo
ti = (0, T)  # Rango de s en el que resolver las ecuaciones de rayo

show(L)

# Resuelve el sistema de ODEs

# method = 'Radau'  # método numérico de solución de la ODE
method = 'RK45'
# method = 'BDF'

# rays = init_rays(NR, k0, L)  # condiciones iniciales
rays = init_rays_parallel(NR, k0, L)  # condiciones iniciales
for i in range(NR):
    f0 = rays[:, i]  # Condiciones iniciales
    sol = solve_ivp(rayeq, ti, f0,
                    dense_output=True,
                    t_eval=t,
                    method=method,
                    args=(l, L, w))

    # Gráfica de las soluciones
    plt.plot(sol.y[0], sol.y[1], lw=0.5)  # sol.t contiene los valores de tiempo, sol.y[0] contiene los valores de y1

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Trayectorias.png', dpi=2000)

plt.show()
