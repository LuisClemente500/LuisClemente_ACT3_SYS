"""
Actividad formativa 3 – Filtros digitales
Código sencillo con SciPy y Matplotlib
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 1. Señal de prueba: mezcla de dos tonos + ruido blanco
fs = 5000            # Hz, frecuencia de muestreo
t  = np.linspace(0, 1.0, fs, endpoint=False)
sig = (np.sin(2*np.pi*100*t)        # tono grave 100 Hz
       + 0.5*np.sin(2*np.pi*1000*t) # tono agudo 1 kHz
       + 0.3*np.random.randn(len(t)))  # ruido

# 2. Diseño de tres filtros Butterworth (orden bajo para simplicidad)
def butter_filter(ftype, cutoff, fs, order=4):
    nyq = 0.5*fs
    if ftype == "band":
        normal = [f/nyq for f in cutoff]  # par de bordes
    else:
        normal = cutoff/nyq
    b, a = signal.butter(order, normal, btype=ftype, analog=False)
    return b, a

b_lp, a_lp   = butter_filter("low",   300,          fs)        # pasa-bajos
b_hp, a_hp   = butter_filter("high",  500,          fs)        # pasa-altos
b_bp, a_bp   = butter_filter("band",  [300, 700],   fs)        # pasa-bandas

# 3. Filtrado (filtfilt evita desfase)
sig_lp = signal.filtfilt(b_lp, a_lp, sig)
sig_hp = signal.filtfilt(b_hp, a_hp, sig)
sig_bp = signal.filtfilt(b_bp, a_bp, sig)

# 4. Graficar señales antes y después (solo 0.05 s para ver claro)
plt.figure(figsize=(8,6))
plt.subplot(4,1,1); plt.title("Original");  plt.plot(t[:250], sig[:250])
plt.subplot(4,1,2); plt.title("Low-pass 300 Hz");  plt.plot(t[:250], sig_lp[:250])
plt.subplot(4,1,3); plt.title("High-pass 500 Hz"); plt.plot(t[:250], sig_hp[:250])
plt.subplot(4,1,4); plt.title("Band-pass 300-700 Hz"); plt.plot(t[:250], sig_bp[:250])
plt.tight_layout(); plt.show()

# 5. Respuesta en frecuencia de un filtro (ejemplo: pasa-bajos)
w, h = signal.freqz(b_lp, a_lp, fs=fs)
plt.figure(); plt.semilogx(w, 20*np.log10(abs(h))); plt.title("Respuesta en frecuencia LP 300 Hz")
plt.xlabel("Frecuencia (Hz)"); plt.ylabel("Magnitud (dB)"); plt.grid(); plt.show()
