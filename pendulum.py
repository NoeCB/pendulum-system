import sys
import numpy as np
from collections import deque

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton, 
                             QGroupBox, QFrame)
from PyQt5.QtCore import QTimer, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }
QGroupBox {
    color: white; font-weight: bold; border: 1px solid #555;
    border-radius: 5px; margin-top: 10px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
QLabel { color: #cccccc; font-size: 12px; }
QDoubleSpinBox {
    background-color: #404040; color: #00e676; border: 1px solid #555;
    padding: 3px; border-radius: 3px;
}
QPushButton {
    background-color: #0d47a1; color: white; border: none;
    padding: 8px; border-radius: 4px; font-weight: bold;
}
QPushButton:hover { background-color: #1565c0; }
QPushButton#btnStop { background-color: #b71c1c; }
QPushButton#btnStop:hover { background-color: #d32f2f; }
QPushButton#btnClear { background-color: #f57f17; }
"""

def derivatives(state, t, L1, L2, m1, m2, g):
    theta1, z1, theta2, z2 = state
    
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    den = (m1 + m2) * L1 - m2 * L1 * c * c

  
    dz1 = (m2 * g * np.sin(theta2) * c \
           - m2 * s * (L1 * z1**2 * c + L2 * z2**2) \
           - (m1 + m2) * g * np.sin(theta1)) / (L1 * den / L1)
    
    den1 = L1 * (2*m1 + m2 - m2*np.cos(2*theta1 - 2*theta2))
    den2 = L2 * (2*m1 + m2 - m2*np.cos(2*theta1 - 2*theta2))

    term1_num = -g*(2*m1 + m2)*np.sin(theta1) - m2*g*np.sin(theta1 - 2*theta2) \
                - 2*s*m2*(z2**2*L2 + z1**2*L1*c)
    dz1_exact = term1_num / den1

    term2_num = 2*s*(z1**2*L1*(m1 + m2) + g*(m1 + m2)*np.cos(theta1) \
                + z2**2*L2*m2*c)
    dz2_exact = term2_num / den2

    return np.array([z1, dz1_exact, z2, dz2_exact])

def rk4_step(state, dt, L1, L2, m1, m2, g):
    
    k1 = derivatives(state, 0, L1, L2, m1, m2, g)
    k2 = derivatives(state + 0.5 * dt * k1, 0, L1, L2, m1, m2, g)
    k3 = derivatives(state + 0.5 * dt * k2, 0, L1, L2, m1, m2, g)
    k4 = derivatives(state + dt * k3, 0, L1, L2, m1, m2, g)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#2b2b2b')
        self.axes = self.fig.add_subplot(111)
        
        self.axes.set_facecolor('#2b2b2b')
        self.axes.tick_params(colors='white', labelbottom=False, labelleft=False)
        self.axes.grid(False) 
        
      
        for spine in self.axes.spines.values():
            spine.set_edgecolor('#555555')

        super(MplCanvas, self).__init__(self.fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador Doble Péndulo")
        self.resize(1000, 700)
        self.setStyleSheet(STYLESHEET)

        self.state = np.array([np.pi/2, 0, np.pi/2, 0]) 
        
        self.dt = 0.02
        self.trace_len = 300 
        self.x2_trace = deque(maxlen=self.trace_len)
        self.y2_trace = deque(maxlen=self.trace_len)

        self.timer = QTimer()
        self.timer.setInterval(15)
        self.timer.timeout.connect(self.update_simulation)

        self.setup_ui()
        
        self.init_plot()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Panel Izquierdo ---
        left_panel = QFrame()
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)

        grp_params = QGroupBox("Propiedades Físicas")
        form_layout = QVBoxLayout()
        
        self.spin_m1 = self.crear_input("Masa 1 (kg):", 1.0, form_layout)
        self.spin_m2 = self.crear_input("Masa 2 (kg):", 1.0, form_layout)
        self.spin_l1 = self.crear_input("Longitud 1 (m):", 1.0, form_layout)
        self.spin_l2 = self.crear_input("Longitud 2 (m):", 1.0, form_layout)
        self.spin_g  = self.crear_input("Gravedad (m/s²):", 9.8, form_layout)
        
        grp_params.setLayout(form_layout)
        left_layout.addWidget(grp_params)

    
        self.btn_start = QPushButton("▶ Iniciar")
        
       
        self.btn_start.clicked.connect(self.timer.start)
        
        self.btn_stop = QPushButton("⏸ Pausar")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.clicked.connect(self.timer.stop)

        self.btn_reset = QPushButton("↺ Reiniciar (Arriba)")
        self.btn_reset.setObjectName("btnClear")
        self.btn_reset.clicked.connect(self.reset_simulation)

        left_layout.addSpacing(20)
        left_layout.addWidget(self.btn_start)
        left_layout.addWidget(self.btn_stop)
        left_layout.addWidget(self.btn_reset)
        left_layout.addStretch()

        self.canvas = MplCanvas(self)
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.canvas)

    def crear_input(self, texto, valor, layout):
        lbl = QLabel(texto)
        layout.addWidget(lbl)
        spin = QDoubleSpinBox()
        spin.setRange(0.1, 50.0)
        spin.setSingleStep(0.1)
        spin.setValue(valor)
        layout.addWidget(spin)
        return spin

    def init_plot(self):
       
        self.trace_line, = self.canvas.axes.plot([], [], '-', lw=1, color='#00e676', alpha=0.5)
     
        self.rods_line, = self.canvas.axes.plot([], [], 'o-', lw=3, color='white', markersize=8)
     
        L_total = self.spin_l1.value() + self.spin_l2.value()
        limit = L_total * 1.2
        self.canvas.axes.set_xlim(-limit, limit)
        self.canvas.axes.set_ylim(-limit, limit)
        self.canvas.axes.set_aspect('equal') 
        self.canvas.draw()

    def reset_simulation(self):
        self.timer.stop()
       
        self.state = np.array([np.pi/2, 0, np.pi/2, 0])
        self.x2_trace.clear()
        self.y2_trace.clear()
        self.update_plot_limits() 
        self.update_simulation() 

    def update_plot_limits(self):
        L_total = self.spin_l1.value() + self.spin_l2.value()
        limit = L_total * 1.2
        self.canvas.axes.set_xlim(-limit, limit)
        self.canvas.axes.set_ylim(-limit, limit)

    def update_simulation(self):
        # 1. Leer parámetros
        m1 = self.spin_m1.value()
        m2 = self.spin_m2.value()
        l1 = self.spin_l1.value()
        l2 = self.spin_l2.value()
        g  = self.spin_g.value()

     
        for _ in range(4):
            self.state = rk4_step(self.state, self.dt/4, l1, l2, m1, m2, g)

        theta1, _, theta2, _ = self.state
        
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)

      
        self.x2_trace.append(x2)
        self.y2_trace.append(y2)


        self.rods_line.set_data([0, x1, x2], [0, y1, y2])
        
        self.trace_line.set_data(list(self.x2_trace), list(self.y2_trace))

        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
