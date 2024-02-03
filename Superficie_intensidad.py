import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.special import gamma
import mpmath
from scipy import integrate

# Entrada de constantes de parámetros

# [W] Potencia del láser
P = 500 * (10 ** 12)

# [m] Longitud de onda del láser
lam = 350 * (10 ** (-9))

# [m] Radio del lente del láser
ro = 0.4

# [m] Distancia del foco al centro blanco
FB = 2000 * (10 ** (-6))

# [m] Radio del blanco
Rb = 1100 * (10 ** (-6))

# [rad] Semiángulo del foco
alfa = np.radians(8)

# [m] Distancia del foco a la lente del haz
Zf = ro / np.tan(alfa)

# [m] Radio de intersección entre el blanco y el láser (¿la interpretación del signo -?)
rf = np.tan(alfa) * ((FB + np.sqrt((Rb ** 2) * (1 + (np.tan(alfa)) ** 2) - (FB ** 2) * ((np.tan(alfa)) ** 2))) / (
        1 + (np.tan(alfa)) ** 2))

# Semiángulo de incidencia sobre el blanco
gama = np.degrees(np.arcsin(rf / Rb))

# Ángulo de incidencia sobre el blanco
gam = 2 * gama

# Ángulo focal máximo dado Rb y FB
alfamax = np.degrees(np.arctan(Rb / FB))

# [m] Distancia máxima entre borde de iluminación y lente
Zmax = (ro / np.tan(alfa)) - (FB + np.sqrt(Rb ** 2 - rf ** 2))

# [m] Distancia del primer contacto de iluminación entre el blanco y lente
Zmin = (ro / np.tan(alfa)) - (FB + Rb)

# [mm] Altura del casquete
H = (Zmax - Zmin) * 10 ** 6

# [m] Distancia media del láser al punto de incidencia del blanco
Z = (Zmax - Zmin) / 2

# [m] Waist radius PARÁMETRO MUY IMPORTANTE
wo = 0.3 * (10 ** (-3))

# [m] Distancia Rayleigh
Zr = (np.pi * (wo ** 2)) / lam

# [m]
w = wo * np.sqrt(1 + (Z / Zr) ** 2)

# Exponente de la exponencial de la expresión de la intensidad
n = 4

# Cálculo de Io suponiéndola constante para r = rf y w ~ wo
o = 2 / n
p = 2 * ((ro / w) ** n)

gamma_incompleta = np.vectorize(mpmath.gammainc)(o, p)
gamma_value = gamma(2 / n)

Io = (P * n) / (np.pi * (w ** 2) * (2 ** (1 - (2 / n))) * (gamma_value - gamma_incompleta))

print("La distancia de Rayleigh es: ", Zr, "m")
print("La intensidad máxima es: ", Io, "W/m2")
print("El ángulo focal máximo (2*alfa) para una FB y Rb dada es de: {:.2f}".format(alfamax * 2))
print("El ángulo de incidencia (2*gamma) sobre el blanco es de: {:.2f}".format(gama * 2))
print("El radio de intersección (rf) entre el haz y el blanco es de: {:.2f} mu m".format(rf * 10 ** 6))
print("Distancia entre el borde de iluminación y el lente del láser es de: {:.8f} m".format(Zmax))
print("Distancia del primer contacto de iluminación entre el blanco y el lente del láser es de: {:.8f} m".format(Zmin))
print("Altura del casquete: {:.2f} mu m".format(H))
print("El valor de w(z) es de: ", w * 10 ** 6, "mu m")
print("La distancia del foco al lente del haz es: ", Zf, "m")


# Define la función en coordenadas cilíndricas
def cylindrical_function(theta, r):
    # Cálculo intermedio de la función gamma incompleta para el valor de n
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = Io * np.exp(-2 * ((r / w) ** n))  # Función de intensidad
    return x, y, z


# Genera datos para theta y r
theta = np.linspace(0, 2 * np.pi, 1000)
r = np.linspace(0, rf, 1000)  # Limitamos r desde 0 a rf
theta, r = np.meshgrid(theta, r)

# Calcula las coordenadas cilíndricas
x, y, z = cylindrical_function(theta, r)

# Crea la figura y los ejes 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot de la superficie cilíndrica con colores personalizados según Z
surf = ax.plot_surface(x * 10 ** 6, y * 10 ** 6, z * (10 ** (-13)), cmap='plasma', alpha=0.8)

# Configuración de ejes
ax.set_xlabel('r [$\mathbf{\mu}$m]', fontsize=12, fontweight='bold')
ax.set_ylabel('r [$\mathbf{\mu}$m]', fontsize=12, fontweight='bold')

# Utiliza ScalarFormatter para los ejes x, y, y z con 4 cifras significativas
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 4))  # Limita a 4 cifras significativas
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)

ax.set_title('Distribución de la intensidad del haz láser \n sobre la superficie del blanco esférico', fontsize=16,
             fontweight='bold')

# Ajusta la perspectiva de los ángulos de elevación y azimut
ax.view_init(elev=20, azim=30)

# Agrega la barra de colores con 4 cifras significativas
colorbar = fig.colorbar(surf, ax=ax, pad=0.1, label='Intensidad (GW/cm$^2$', format='%.4e')
colorbar.set_label('Intensidad (GW/cm$\mathbf{^2}$)', fontsize=12, fontweight='bold')

# Ajusta el tamaño de la gráfica
fig.set_size_inches(10, 6)

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Distribución de intensidad.png', dpi=1000)

# Muestra la gráfica
plt.show()

#############################################################

# Gráfica de la independencia de Z en la intensidad en la superificie del blanco

# Definir los valores de r y Z
r_values = np.linspace(0, Rb, 1000)  # Valor de r de acuerdo al intervalo de Z
Z_values = np.linspace(Zmin, Zmin + Rb, 1000)  # La mayor diferencia entre Z sería zmin y zmin+Rb

# Crear matrices 2D de r y Z
r, Z = np.meshgrid(r_values, Z_values)


# Definir la función I(r, Z). Aquí se supone una Io constante para w ~ wo
def intensity_function(r, Z):
    # Cálculo de Io
    wo_z = wo * np.sqrt(1 + (Z / Zr) ** 2)
    o = 2 / n
    gamma_value = gamma(o)
    Io = (P * n) / (np.pi * (wo_z ** 2) * (2 ** (1 - (2 / n))) * gamma_value)
    result = Io * np.exp(-2 * ((r / wo_z) ** n))

    return result


# Calcular los valores de intensidad para cada par (r, Z)
I_values = intensity_function(r, Z)

# Crear un gráfico de contorno
plt.figure(figsize=(10, 6))
contour_plot = plt.contour(r * 10 ** 6, Z, I_values, cmap='viridis', levels=500)

# Etiquetas y título con negrita y tamaño de fuente aumentado
plt.xlabel('r ($\mathbf{\mu}$m)', fontweight='bold', fontsize=12)
plt.ylabel('Z (m)', fontweight='bold', fontsize=12)
plt.title('Valor de la intensidad', fontweight='bold', fontsize=16)

# Agregar una barra de colores con negrita y tamaño de fuente aumentado
cbar = plt.colorbar(contour_plot, label='Intensidad (GW/cm$\mathbf{^2}$', format=ScalarFormatter())
cbar.update_ticks()

cbar.set_label('Intensidad (GW/m$^2$)', fontweight='bold', fontsize=12)

# Configuración de la cuadrícula con líneas más transparentes
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Independencia de Z.png', dpi=1200)

# Muestra el gráfico de contorno
plt.show()

#############################################################

# Distribución de intensidad del haz en en el lente (Z = 0) y blanco. Depende fundamentalmente de wo

# [m] Distancia del láser al punto de incidencia del blanco
Z1 = 0

# [m]
w1 = wo * np.sqrt(1 + (Z1 / Zr) ** 2)

# Definir la función lambda
funcion1 = lambda r: 100 * np.exp(- 2 * ((r / w1) ** n))

# Crear un rango de valores r
r_values = np.linspace(-ro, ro, 1000)

# Calcular los valores y aplicando la función lambda
y_values = funcion1(r_values)

# Graficar la función con negrita y tamaño de fuente aumentado
plt.plot(r_values, y_values)
plt.xlabel('r (m)', fontweight='bold', fontsize=12)
plt.ylabel('I (W/m$^2$) en tanto %', fontweight='bold', fontsize=12)
plt.title(r'$\mathbf{Distribución \ de \ intensidad \ del \ láser \ en \ Z = 0 \ m}$', fontsize=14, fontweight='bold')
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Distribución de intensidad en Z=0.png', dpi=1000)

# Muestra la gráfica
plt.show()

# ---------------------------------------------------------#

# Distribución de intensidad para la superficie de incidencia del blanco

# Definir la función lambda
funcion2 = lambda r, Zv: 100 * np.exp(- 2 * ((r / (wo * np.sqrt(1 + (Zv / Zr) ** 2))) ** n))

# Crear un rango de valores x
r_values = np.linspace(-rf, rf, 599)

# Rango de valores para Z
z_values1 = np.linspace(Zmin, Zmax, 300)
z_values2 = z_values1[::-1]
z_values2 = z_values2[:-1]

z_values = np.concatenate((z_values2, z_values1))  # Concatenación de los valores de Z para que en -rf sea Zmax y
# hasta rf vuelva a ser Zmax

np.savetxt('zvalues.txt', z_values, header='Valores de Z', delimiter='\t', fmt='%f', comments='')

# Calcular los valores de I(r,Z)
y_values = funcion2(r_values, z_values)

# Graficar la función con negrita y tamaño de fuente aumentado
plt.plot(r_values * 10 ** 6, y_values)
plt.xlabel('r ($\mathbf{\mu}$m)', fontweight='bold', fontsize=12)
plt.ylabel('I (W/m$^2$) en tanto %', fontweight='bold', fontsize=12)
plt.title(r'$\mathbf{Distribución \ de \ intensidad \ del \ láser \ en \ el \ blanco}$', fontsize=14, fontweight='bold')
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Distribución de intensidad en el blanco.png', dpi=1000)

# Muestra la gráfica
plt.show()

#############################################################

# Cálculo de energía del haz en una sección circular de radio rp

# Número de rayos = nodos
numray = 600

# Radio de la circunferencia plana sobre la que se plantea su cálculo de energía
rp = rf

# Cálculo de w según la Z en donde se encuentre rp
Z = Zf - (rp / np.tan(alfa))

w = wo * np.sqrt(1 + (Z / Zr) ** 2)

p = 2 * ((rp / w) ** n)

gamma_incompleta = np.vectorize(mpmath.gammainc)(o, p)

Io = (P * n) / (np.pi * (w ** 2) * (2 ** (1 - (2 / n))) * (gamma_value - gamma_incompleta))

# Define la función a integrar
f = lambda r: (Io / P) * np.exp(- 2 * ((r / w) ** n)) * 2 * np.pi * r

# Límites de integración
a, b = 0, rp

# Calcular la integral real
f_real = integrate.quad(f, a, b)


# -------------Cálculo integral mediante Simpson-----------------------#

# Método de Simpson
def simpson_integration(a, b, numray, f):
    h = (b - a) / (numray - 1)

    x_values = np.linspace(a, b, numray)
    y_values = f(x_values)

    integral = h / 3 * (y_values[0] + 4 * np.sum(y_values[1:-1:2]) + 2 * np.sum(y_values[2:-2:2]) + y_values[-1])

    return integral
# a límite inferior de la integral, b el superior, numray el número de nodos y f la función a integrar


# Calcular la integral utilizando la regla de Simpson
Q_sum = simpson_integration(a, b, numray, f)

Error = (np.abs(f_real[0] - Q_sum) / f_real[0]) * 100

print("El valor de la integral entre", a, "y", b, "por Simpson es: ", Q_sum)
print("El valor de la integral real entre", a, "y", b, "es: ", f_real[0])
print("El error de aproximación es: ", Error, "%")

# --------Cálculo de delta para obtener rayos monoenergéticos-------#


#############################################################
# Gráfica del valor de las diferencias de gamma

# Definir valores de n
n_values = np.arange(2, 13, 2)

# Rango de valores para r
r_values = np.linspace(wo, Rb, 600)  # el valor mínimo es el waist radius y el máx ro

# Configurar el gráfico
plt.figure(figsize=(8, 5))
plt.grid(color='gray', linestyle='--', linewidth=0.5)

for n in n_values:
    gamma_graf = np.zeros((1, 600))

    for i in range(len(r_values)):
        j = r_values[i]
        o = 2 / n
        p = 2 * ((j / wo) ** n)
        gamma_graf[0, i] = gamma(o) - mpmath.gammainc(o, p)

    plt.plot(r_values, gamma_graf.flatten(), label=f'n={n}')

# Agregar títulos y leyenda
plt.title(r'$\mathbf{Valor \ de \ \Gamma (2/n) - \Gamma (2/n, 2(r/w)^n)}$', fontsize=14, fontweight='bold')
plt.xlabel('Distancia radial r (m)', fontweight='bold', fontsize=12)
plt.legend()

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Diferencia_gammas.png', dpi=1000)

plt.show()

# ----------------------------------------------------------------#

# Valor de Io para w = wo porque para el rango de valores de Z w ~ wo

gamma_grafn = np.zeros((1, 600))
r_values = np.linspace(0, rf, 600)  # Valores de Io entre wo y rf

# Distintos valores de la diferencia de gammas en función de r
o = 2 / n

for i in range(len(r_values)):
    j = r_values[i]
    if j < wo:
        p = 2 * ((wo / w) ** n)
    elif j >= wo:
        p = 2 * ((j / w) ** n)
    gamma_grafn[0, i] = gamma(o) - mpmath.gammainc(o, p)

# Valores de Io para distintos r
Int_grafn = (P * n) / ((2 ** (1 - (2 / n))) * np.pi * (w ** 2) * gamma_grafn)

print("Diferencia entre el mínimo y el máximo valor de Io", (Int_grafn[0, 0] - Int_grafn[0, 599]) * (10 ** -13))

# Se guarda fichero txt
np.savetxt('Io.txt', Int_grafn, header='Matriz de vectores', delimiter='\t', fmt='%f', comments='')

plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.title(r'$\mathbf{Valor \ de \ I_0}$',
          fontsize=14, fontweight='bold')

# Agregar títulos a los ejes x e y
plt.xlabel('Distancia radial r (m)', fontweight='bold', fontsize=12)
plt.ylabel('$\mathbf{I_0}$ (GW/cm$^2$)', fontweight='bold', fontsize=12)

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Valor de I0.png', dpi=1000)

plt.xlim([0, rf])
plt.plot(r_values, Int_grafn.flatten() * (10 ** -13))

plt.show()

############################################################################

# Cálculo de vectores como número de haces

# Radio del lente
ro = ro

# Longitud entre nodos
h = (2 * ro) / (numray - 1)

k = np.zeros((numray, 2))
for i in range(0, numray):
    k[i, 0] = -Zf
    k[i, 1] = ro - (h * i)

# Matriz rayos: x y kx ky
matrixR = np.zeros((numray, 4))
for i in range(0, numray):
    matrixR[i, 0] = 0
    matrixR[i, 1] = -ro + h * i
    matrixR[i, 2] = Zf
    matrixR[i, 3] = ro - (h * i)

# Se guarda fichero txt de los vectores
np.savetxt('matriz_k_vectores.txt', matrixR, header='X\t\tY\t\tKx\t\tKy'
           , delimiter='\t', fmt='%f', comments='')

# Graficar los vectores
fig, ax = plt.subplots()
for i in range(numray):
    ax.quiver(0, ro - (i * h), -k[i, 0], -k[i, 1], angles='xy', scale_units='xy', scale=1, color='r', width=0.0002)

# Configurar el gráfico
ax.set_xlim([-0.2, 3])  # Límites del eje X (distancia del lente al foco)
ax.set_ylim([-0.6, 0.6])  # Límites del eje Y (eje del radio del lente)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(color='gray', linestyle='--', linewidth=0.5)
ax.set_title(r'$\mathbf{Rayos \ del \ lente. \ Para \ m \ = \ ' + str(numray) + ' \ rayos}$', fontsize=14,
             fontweight='bold')
# Agregar títulos a los ejes x e y
ax.set_xlabel('Distancia lente-foco (m)', fontweight='bold', fontsize=12)
ax.set_ylabel('Distancia radial del lente (m)', fontweight='bold', fontsize=12)

# Invertir el eje X
ax.invert_xaxis()

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Haces de rayos.png', dpi=2000)

plt.show()
