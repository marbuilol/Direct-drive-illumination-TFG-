import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- Método de Simpson ------------------------- #
def simpson_integration(a, b, particiones, f):
    h = (b - a) / (particiones - 1)

    x_values = np.linspace(a, b, particiones)
    y_values = f(x_values)

    integral = h / 3 * (y_values[0] + 4 * np.sum(y_values[1:-1:2]) + 2 * np.sum(y_values[2:-2:2]) + y_values[-1])

    return integral


# --------Cálculo de deltas para obtener rayos monoenergéticos-------#

seminumray = 100  # Número de rayos desde "a" a "b"
liminf = 0 # Límite inferior de integración
limsup = 4.e-3  # Límite superior de integración
sigma = 2
valueexp = 2  # Valor del exponente
Io = 1.e+5  # Valor de la intensidad cuando r = 0
coef = Io * 2 * np.pi
trozos = 1000  # Particiones en el método de Simpson
err = 1.e-7  # Margen de error

Rl = 10  # Radio del lente
Rs = limsup  # Radio de la sección de estudio
alfa = np.radians(8)  # Semiángulo del foco

Fb = Rs / np.tan(alfa)  # Distancia de la sección de estudio al foco
Fl = Rl / np.tan(alfa)  # Distancia del lente al foco
H = Fl - Fb  # Distancia entre lente y la sección de estudio
numray = seminumray * 2  # Cantidad total de "r"

func = lambda r: coef * r * (np.exp(-((r / sigma) ** valueexp)))  # Función a integrar

totalenergy = simpson_integration(liminf, limsup, trozos, func)  # Energía de la sección de estudio

print("La energía del láser en un intervalo t es: ", totalenergy)

rayenergy = totalenergy / seminumray  # Energía de cada rayo

print("La energía de cada rayo es: ", rayenergy)

# Número de radios desde a hasta b
rad = np.zeros(seminumray + 1)  # seminumray columnas

func_g = np.zeros(seminumray)  # Vector diferencia de energías

for i in range(seminumray):
    radprueba = rad[i] + limsup / 20
    func_g[i] = np.abs(simpson_integration(rad[i], radprueba, trozos, func) - rayenergy)
    while func_g[i] >= err:
        radprueba = radprueba - (simpson_integration(rad[i], radprueba, trozos, func) - rayenergy) / func(radprueba)
        func_g[i] = np.abs(simpson_integration(rad[i], radprueba, trozos, func) - rayenergy)

    if radprueba <= limsup:
        rad[i + 1] = radprueba

print("Los radios desde 'a' hasta 'b' son: ", rad)
print("La diferencia de energía entre cada intervalo de radios son: ", func_g)

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

# --------------------- Gráfica de los vectores -------------------------------#
fig, ax = plt.subplots()
for i in range(numray):
    ax.quiver(matrixR[i, 0], matrixR[i, 1], matrixR[i, 2], matrixR[i, 3], angles='xy', scale_units='xy', scale=1,
              color='r', width=0.001)

# Configurar límites del gráfico y estilos de líneas
ax.set_xlim([H - Fb / 10, Fl + Fb / 10])  # Límites del eje X (distancia del lente al foco)
ax.set_ylim([-Rs - Rs / 20, Rs + Rs / 20])  # Límites del eje Y (eje del radio del lente)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(color='gray', linestyle='--', linewidth=0.5)
ax.set_title(r'$\mathbf{Rayos \ monoenergéticos. \ Para \ ' + str(seminumray) + '\ rayos}$', fontsize=14,
             fontweight='bold')

# Agregar títulos a los ejes x e y
ax.set_xlabel('Distancia sección-lente (m)', fontweight='bold', fontsize=12)
ax.set_ylabel('Distancia radial (m)', fontweight='bold', fontsize=12)

# Invertir el eje X
ax.invert_xaxis()

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('RayosMonoEnergéticos.png', dpi=2000)

plt.show()

# ----------------------- Gráfica de la función a integrar -----------------------#

# Crear un array de valores de r para el plot
r_values = np.linspace(0, Rs, 1000)  # Ajusta el límite superior según sea necesario

# Calcular los valores de la función para los valores de r
y_values = func(r_values)

# Realizar el plot de la función
plt.plot(r_values, y_values, label='$\mathbf{f(r)}$')

# Configuraciones adicionales del plot
plt.title('Gráfica de la función a integrar', fontweight='bold', fontsize=12)
plt.xlabel('r (m)', fontweight='bold', fontsize=10)
plt.ylabel('f(r)', fontweight='bold', fontsize=10)
plt.legend()
plt.grid(True)

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Gráfica a integrar rayos monoenergéticos.png', dpi=2000)

plt.show()

# ----------------------- Gráfica de energías de cada rayo -----------------------#
energiasderayos = np.zeros(seminumray)
for i in range(seminumray):
    energiasderayos[i] = simpson_integration(rad[i], rad[i + 1], trozos, func)

posiciondelrayo = list(range(1, seminumray + 1))

# Realizar el plot de la función
plt.plot(posiciondelrayo, energiasderayos)

# Configuraciones adicionales del plot
plt.title('Energía de cada rayo', fontweight='bold', fontsize=12)
plt.xlabel('Rayo nº', fontweight='bold', fontsize=10)
plt.ylabel('E (MJ)', fontweight='bold', fontsize=10)
plt.grid(True)

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom en la imagen descargada
plt.savefig('Gráfica de energías.png', dpi=2000)

plt.show()
