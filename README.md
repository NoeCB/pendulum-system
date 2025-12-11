# ⚛️ Simulador de Péndulo Doble (Python + PyQt5)

Este proyecto es una aplicación de escritorio interactiva que simula y visualiza el movimiento caótico de un **péndulo doble** en tiempo real. Utiliza las ecuaciones de Lagrange y el método numérico Runge-Kutta 4 (RK4) para resolver la física del sistema.

![Estado del Proyecto](https://img.shields.io/badge/Estado-Terminado-green)
![Python](https://img.shields.io/badge/Python-3.x-blue)

## 📋 Características

* **Simulación en Tiempo Real:** Visualización fluida a ~60 FPS.
* **Parámetros Ajustables:** Modifica en vivo la masa y longitud de ambos péndulos, así como la gravedad.
* **Traza de Movimiento:** Dibuja la estela del segundo péndulo para visualizar el caos y los patrones geométricos.
* **Interfaz Gráfica Profesional:** Panel de control lateral y modo oscuro integrado.
* **Motor Físico:** Implementación manual de RK4 para alta precisión numérica.

## 🛠️ Requisitos e Instalación

Para ejecutar este simulador, necesitas tener instalado **Python 3**.

### 1. Clonar o descargar
Descarga el archivo `pendulum.py` (o clona este repositorio si usas git).

### 2. Instalar dependencias
El proyecto utiliza `PyQt5` para la ventana, `Matplotlib` para el gráfico y `NumPy` para los cálculos. Instálalos ejecutando:

```bash
pip install numpy matplotlib PyQt5
