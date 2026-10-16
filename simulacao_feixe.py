"""
Simulação de um feixe laser gaussiano (modo TEM00) com matplotlib.

Equações usadas:
    z_R  = pi * w0^2 / lambda               (comprimento de Rayleigh)
    w(z) = w0 * sqrt(1 + (z / z_R)^2)       (raio do feixe)
    I(r,z) = I0 * (w0/w(z))^2 * exp(-2 r^2 / w(z)^2)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------- Parâmetros ---------------------------------
comprimento_onda = 633e-9   # m  (laser He-Ne vermelho)
w0 = 0.5e-3                 # m  (cintura do feixe)
z_amostra = None            # m  (None -> usa 1 * z_R para o perfil transversal)

# ----------------------------- Funções ------------------------------------
def rayleigh(w0, lam):
    return np.pi * w0**2 / lam

def raio_feixe(z, w0, zR):
    return w0 * np.sqrt(1 + (z / zR) ** 2)

def intensidade(x, z, w0, zR):
    """Intensidade normalizada (I0 = 1) em função da posição transversal x e de z."""
    w = raio_feixe(z, w0, zR)
    return (w0 / w) ** 2 * np.exp(-2 * x**2 / w**2)

# ----------------------------- Nova Função --------------------------------
def salvar_feixe(I_xy, lim, cmap, nome_arquivo="feixe_isolado.png"):
    """
    Salva apenas a imagem do feixe simulado (perfil transversal 2D) 
    sem eixos, barras de cores, rótulos ou margens em branco.
    """
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#101010")
    ax.set_facecolor("#181818")
    
    # Remove os eixos e molduras para focar puramente no feixe
    ax.axis('off')
    
    ax.imshow(
        I_xy, extent=[-lim * 1e3, lim * 1e3, -lim * 1e3, lim * 1e3],
        origin="lower", cmap=cmap, vmin=0, vmax=I_xy.max(),
    )
    
    # Salva recortando perfeitamente a área da imagem sem bordas
    plt.savefig(nome_arquivo, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ----------------------------- Cálculos -----------------------------------
zR = rayleigh(w0, comprimento_onda)
if z_amostra is None:
    z_amostra = zR

# Perfil transversal 2D em z = z_amostra
w_s = raio_feixe(z_amostra, w0, zR)
lim = 3 * w_s
xs = np.linspace(-lim, lim, 500)
XX, YY = np.meshgrid(xs, xs)
I_xy = (w0 / w_s) ** 2 * np.exp(-2 * (XX**2 + YY**2) / w_s**2)

# ----------------------------- Colormap "laser" ---------------------------
laser_cmap = LinearSegmentedColormap.from_list(
    "laser", ["black", "#3b0000", "#b00000", "#ff2a00", "#ff9a8a", "white"]
)

# ----------------------------- Chamada da Função --------------------------
# Salva somente o feixe isolado
salvar_feixe(I_xy, lim, laser_cmap, "feixe_isolado.png")

# ----------------------------- Gráficos Originais -------------------------
txt = "white"
fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#101010",
                               gridspec_kw={"wspace": 0.3})

# (1) Perfil transversal 2D
im = ax2.imshow(
    I_xy, extent=[-lim * 1e3, lim * 1e3, -lim * 1e3, lim * 1e3],
    origin="lower", cmap=laser_cmap, vmin=0, vmax=I_xy.max(),
)
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ax2.set_title(f"Perfil transversal em z = {z_amostra*1e3:.0f} mm")
cb = fig.colorbar(im, ax=ax2, label="Intensidade (I/I$_0$)")

# (2) Cortes 1D em diferentes z
xr = np.linspace(-3e-3, 3e-3, 800)
for fator, cor in zip([0, 1, 2, 3], ["#ff2a00", "#ffa500", "#7CFC00", "#00bfff"]):
    zi = fator * zR
    ax3.plot(xr * 1e3, intensidade(xr, zi, w0, zR), color=cor,
             label=f"z = {fator}·$z_R$ = {zi*1e3:.0f} mm")
ax3.set_xlabel("x (mm)")
ax3.set_ylabel("Intensidade (I/I$_0$)")
ax3.set_title("Perfil de intensidade ao longo da propagação")
ax3.grid(alpha=0.25)
ax3.legend(facecolor="#202020", labelcolor=txt, edgecolor="#555")

# Estilo escuro
for ax in (ax2, ax3):
    ax.set_facecolor("#181818")
    ax.tick_params(colors=txt)
    ax.xaxis.label.set_color(txt)
    ax.yaxis.label.set_color(txt)
    ax.title.set_color(txt)
    for s in ax.spines.values():
        s.set_color("#555")
cb.ax.yaxis.label.set_color(txt)
cb.ax.tick_params(colors=txt)

fig.suptitle(
    rf"Feixe laser gaussiano — $\lambda$ = {comprimento_onda*1e9:.0f} nm, "
    rf"$w_0$ = {w0*1e3:.1f} mm, $z_R$ = {zR:.2f} m",
    color=txt, fontsize=14,
)
plt.savefig("feixe_gaussiano.png", dpi=150, facecolor=fig.get_facecolor())
plt.show()