import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

# Entrada de constantes de parámetros
P = 500 * (10 ** 12)  # [W] Potencia del láser
l = 350 * (10 ** (-9))  # [m] Longitud de onda del láser

ro = 0.9  # [m] Radio del lente del láser
FB = 1400 * (10 ** (-6))  # [m] Distancia del foco al centro blanco
Rb = 1100 * (10 ** (-6))  # [m] Radio del blanco
alfa = np.radians(20)  # [rad] Ángulo del foco

# [m] Radio de intersección entre el blanco y el láser (¿la interpretación del signo +-?)
rf = np.tan(alfa) * ((FB + np.sqrt((Rb ** 2) * (1 + (np.tan(alfa)) ** 2) - (FB ** 2) * ((np.tan(alfa)) ** 2))) / (
        1 + (np.tan(alfa)) ** 2))

# Semiángulo de incidencia sobre el blanco
gamma = np.degrees(np.arcsin(rf / Rb))

# Ángulo de incidencia sobre el blanco
gam = 2 * gamma

# Ángulo focal máximo dado Rb y FB
alfamax = np.degrees(np.arctan(Rb / FB))

# [m] Distancia máxima entre borde de iluminación y lente
Zmax = (ro / np.tan(alfa)) - (FB + np.sqrt(Rb ** 2 - rf ** 2))

# [m] Distancia del primer contacto de iluminación entre el blanco y lente
Zmin = (ro / np.tan(alfa)) - (FB + Rb)

# [mm] Altura del casquete
H = (Zmax-Zmin) * 10 ** 6

Z = (Zmax - Zmin) / 2  # [m] Distancia del láser al punto de incidencia del blanco

wo = 0.6 * (10 ** (-2))  # [m] Waist radius PARÁMETRO MUY IMPORTANTE
Zr = (np.pi * (wo ** 2)) / l  # Distancia Rayleigh

w = wo * np.sqrt(1 + (Z / Zr) ** 2)  # ¿?


print("El ángulo focal máximo (2*alfa) para una FB y Rb dada es de: {:.2f}".format(alfamax * 2))
print("El ángulo de incidencia (2*gamma) sobre el blanco es de: {:.2f}".format(gamma * 2))
print("El radio de intersección (rf) entre el haz y el blanco es de: {:.2f} mu m".format(rf * 10**6))
print("Distancia entre el borde de iluminación y el lente del láser es de: {:.8f} m".format(Zmax))
print("Distancia del primer contacto de iluminación entre el blanco y el lente del láser es de: {:.8f} m".format(Zmin))
print("Altura del casquete: {:.2f} mu m".format(H))
# Define la función en coordenadas cilíndricas
def cylindrical_function(theta, r):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = ((2 * P) / (np.pi * (w ** 2))) * np.exp(- 2 * ((r / w) ** 2))  # Función de intensidad
    return x, y, z


# Genera datos para theta y r
theta = np.linspace(0, 2 * np.pi, 200)
r = np.linspace(0, rf, 200)  # Limitamos r de 0 a rf

theta, r = np.meshgrid(theta, r)

# Calcula las coordenadas cilíndricas
x, y, z = cylindrical_function(theta, r)

# Crea la figura y los ejes 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot de la superficie cilíndrica con colores personalizados según Z
surf = ax.plot_surface(x * 10 ** 6, y * 10 ** 6, z * (10 ** (-4)), cmap='plasma', alpha=0.8 )

# Configuración de ejes
ax.set_xlabel('r [$\mu$m]', fontsize=12, fontweight='bold')
ax.set_ylabel('r [$\mu$m]', fontsize=12, fontweight='bold')
ax.set_zlabel('I [w/cm$^2$]', fontsize=12, fontweight='bold')
ax.set_title('Distribución de la intensidad del haz láser \n sobre la superficie del blanco esférico', fontsize=16, fontweight='bold')

# Ajusta la perspectiva alrededor del eje Y
ax.view_init(elev=30, azim=50)  # Ajusta los ángulos de elevación y azimut

# Agrega la barra de colores con la escala personalizada
colorbar = fig.colorbar(surf, ax=ax, pad=0.08, label='Intensidad (W/cm$^2$)')
colorbar.set_label('Intensidad (W/cm$^2$)', fontsize=12, fontweight='bold')

# Ajusta el tamaño de la gráfica
fig.set_size_inches(8, 6)

# Ajusta la resolución (dpi) para mejorar la calidad al hacer zoom
plt.savefig('Distribución de intensidad.png', dpi=600)  # Puedes cambiar el nombre y formato según tu preferencia

# Muestra la gráfica
plt.show()
