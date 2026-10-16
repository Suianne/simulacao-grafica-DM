"""
Simulação de um feixe laser gaussiano refletido por um espelho deformável
segmentado de 40 atuadores (3 anéis: 8 + 16 + 16 segmentos, sem os "tilt arms").

Fluxo:
    CSV (tensão por segmento) -> deformação da superfície z(x,y)
    -> fase imposta ao feixe  phi = 4*pi*z/lambda  (reflexão, incidência normal)
    -> campo no espelho E = gaussiana * pupila * exp(i*phi)
    -> campo no plano focal de uma lente (transformada de Fourier, via MFT)
    -> intensidade, comparada com a de um espelho plano.

Uso:
    Edite apenas CAMINHO_CSV na função main() e execute:
        python espelho_deformavel.py

Formato do CSV: duas colunas (Segmento, Tensao_V), com os segmentos 1 a 40.

ATENÇÃO: a geometria (raios dos anéis) foi estimada a partir do desenho e as
sensibilidades (CURV_POR_VOLT, COEF_NM_POR_V, V_REPOUSO) são valores de partida.
Ajuste-os com os dados de calibração/datasheet do seu espelho.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Wedge, Circle
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# ============================ GEOMETRIA (mm) ===============================
R_ESPELHO = 6.35                       # diâmetro do espelho = 12.7 mm
R_ANEIS = [2.7, 5.0, R_ESPELHO]        # raio externo de cada anel (estimado do desenho)
N_SETORES = [8, 16, 16]                # segmentos por anel
PRIMEIRO_SEG = [1, 9, 25]              # numeração do 1º segmento de cada anel
N_SEG = 40

# ======================= PARÂMETROS PADRÃO (ajustáveis) ====================
LAMBDA_NM = 633.0        # comprimento de onda (nm)
W_MM = 3.0               # raio 1/e² do feixe no espelho (mm)
PUPILA_MM = 11.5         # diâmetro da pupila (mm)
FOCO_MM = 500.0          # distância focal da lente que forma o plano focal (mm)
V_MIN, V_MAX = 0.0, 300.0   # faixa de tensão (só para a escala de cores)
V_REPOUSO = 150.0        # tensão que corresponde a "sem deformação" (V)
CURV_POR_VOLT = 5e-4     # modelo bimorfo: curvatura por volt (1/(m·V))
COEF_NM_POR_V = 1.0      # modelo local: deformação por volt (nm/V)
SIGMA_MM = 0.6           # modelo local: suavização entre segmentos (mm)

MODELO = "bimorfo"       # "bimorfo" ou "local"
JANELA_UM = None         # meia-largura da janela do plano focal (µm); None = automática

# Malha de simulação no plano do espelho
N_GRID = 512
L_GRID = 13.0e-3         # lado da malha (m); precisa ser > diâmetro do espelho


# ============================ LEITURA DO CSV ===============================
def ler_csv(caminho):
    """Lê o CSV e devolve um vetor com 40 tensões (índice 0 = segmento 1)."""
    df = pd.read_csv(caminho, sep=None, engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"{caminho}: esperado CSV com 2 colunas (Segmento, Tensao_V).")
    seg = df.iloc[:, 0].astype(int).to_numpy()
    volt = df.iloc[:, 1].astype(str).str.replace(",", ".").astype(float).to_numpy()
    if sorted(seg) != list(range(1, N_SEG + 1)):
        raise ValueError(f"{caminho}: o CSV deve conter exatamente os segmentos 1 a {N_SEG}.")
    v = np.zeros(N_SEG)
    v[seg - 1] = volt
    return v


# ======================= GEOMETRIA DOS SEGMENTOS ===========================
def mapa_segmentos(X, Y):
    """Matriz com o número do segmento (1..40) em cada ponto; 0 fora do espelho.
    X, Y em metros. O ângulo é medido a partir do eixo vertical, no sentido
    anti-horário, como na numeração do desenho (segmento 1 à esquerda do eixo Y+)."""
    r = np.hypot(X, Y) * 1e3
    theta = np.mod(np.arctan2(Y, X) - np.pi / 2, 2 * np.pi)
    ids = np.zeros(X.shape, dtype=int)
    r_in = -1.0
    for r_out, n, primeiro in zip(R_ANEIS, N_SETORES, PRIMEIRO_SEG):
        sel = (r > r_in) & (r <= r_out)
        setor = np.floor(theta / (2 * np.pi / n)).astype(int) % n
        ids[sel] = primeiro + setor[sel]
        r_in = r_out
    return ids


# ======================== MODELOS DE DEFORMAÇÃO ============================
def deformacao_bimorfo(dV, ids, dx, curv_por_volt):
    """Espelho bimorfo: a curvatura local é proporcional à tensão aplicada,
        ∇² z = curv_por_volt * (V - V_repouso)   dentro de cada eletrodo,
    resolvida com z = 0 na borda do espelho. Devolve z em metros."""
    dentro = ids > 0
    n = int(dentro.sum())
    indice = -np.ones(ids.shape, dtype=int)
    indice[dentro] = np.arange(n)

    linhas, colunas, valores = [np.arange(n)], [np.arange(n)], [np.full(n, -4.0)]
    for desloc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        vizinho = np.roll(indice, (-desloc[0], -desloc[1]), axis=(0, 1))
        ok = dentro & (vizinho >= 0)      # vizinho fora do espelho => z = 0
        linhas.append(indice[ok])
        colunas.append(vizinho[ok])
        valores.append(np.ones(ok.sum()))
    A = coo_matrix((np.concatenate(valores), (np.concatenate(linhas), np.concatenate(colunas))),
                   shape=(n, n)).tocsc() / dx**2
    z = np.zeros(ids.shape)
    z[dentro] = spsolve(A, curv_por_volt * dV[dentro])
    return z


def deformacao_local(dV, ids, dx, coef_nm_por_v, sigma_mm):
    """Modelo simples: cada segmento empurra a superfície proporcionalmente à
    tensão, com suavização gaussiana (membrana contínua). Devolve z em metros."""
    suave = gaussian_filter(dV, sigma_mm * 1e-3 / dx)
    return coef_nm_por_v * 1e-9 * suave * (ids > 0)


# ============================ ÓPTICA ========================================
def campo_plano_focal(E, x, lam, f, xf):
    """Campo no plano focal (transformada de Fourier) amostrado nos pontos xf,
    usando a transformada matricial (MFT): permite escolher a janela e a
    resolução do plano focal livremente."""
    F = np.exp(-1j * 2 * np.pi / (lam * f) * np.outer(xf, x))
    return F @ E @ F.T * (x[1] - x[0]) ** 2


def simular(volt, args):
    lam = args.lambda_nm * 1e-9
    f = args.foco_mm * 1e-3
    w = args.w_mm * 1e-3
    R_pupila = args.pupila_mm * 1e-3 / 2

    dx = L_GRID / N_GRID
    x = (np.arange(N_GRID) - N_GRID // 2) * dx
    X, Y = np.meshgrid(x, x)
    ids = mapa_segmentos(X, Y)

    # tensão -> mapa de (V - V_repouso) por pixel
    dV = np.where(ids > 0, volt[np.clip(ids - 1, 0, N_SEG - 1)] - args.v_repouso, 0.0)

    if args.modelo == "bimorfo":
        z = deformacao_bimorfo(dV, ids, dx, args.curv)
    else:
        z = deformacao_local(dV, ids, dx, args.coef_nm_v, args.sigma_mm)

    pupila = np.hypot(X, Y) <= R_pupila
    amp = np.exp(-(X**2 + Y**2) / w**2) * pupila          # amplitude gaussiana (I ∝ e^{-2r²/w²})
    inten = amp**2

    z = z - (z * inten).sum() / inten.sum()                # remove o pistão (média ponderada)
    fase = 4 * np.pi * z / lam                             # luz vai e volta: 2z de caminho

    # plano focal
    w_f0 = lam * f / (np.pi * w)                           # cintura ideal no foco
    janela = args.janela_um * 1e-6 if args.janela_um else 6 * w_f0
    xf = np.linspace(-janela, janela, 401)
    E_ref = campo_plano_focal(amp.astype(complex), x, lam, f, xf)
    E_def = campo_plano_focal(amp * np.exp(1j * fase), x, lam, f, xf)
    I_ref = np.abs(E_ref) ** 2
    I_def = np.abs(E_def) ** 2
    pico_ref = I_ref.max()
    I_ref, I_def = I_ref / pico_ref, I_def / pico_ref

    W = 2 * z                                              # erro de frente de onda
    Wm = (W * inten).sum() / inten.sum()
    rms_nm = np.sqrt(((W - Wm) ** 2 * inten).sum() / inten.sum()) * 1e9

    return dict(x=x, ids=ids, z=z, fase=fase, pupila=pupila, xf=xf,
                I_ref=I_ref, I_def=I_def, rms_nm=rms_nm, strehl=I_def.max(), w_f0=w_f0)


# ============================ GRÁFICOS ======================================
CMAP_LASER = LinearSegmentedColormap.from_list(
    "laser", ["black", "#3b0000", "#b00000", "#ff2a00", "#ff9a8a", "white"])
TXT = "white"


def estilo_escuro(ax):
    ax.set_facecolor("#181818")
    ax.tick_params(colors=TXT)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    for s in ax.spines.values():
        s.set_color("#555")


def barra_cores(fig, im, ax, rotulo):
    cb = fig.colorbar(im, ax=ax, label=rotulo, fraction=0.046, pad=0.04)
    cb.ax.yaxis.label.set_color(TXT)
    cb.ax.tick_params(colors=TXT)


def contornos_segmentos(ax, cor="k", lw=0.6, alpha=0.6):
    """Desenha a divisão dos 40 segmentos sobre um eixo em mm."""
    r_in = 0.0
    for r_out, n in zip(R_ANEIS, N_SETORES):
        ax.add_patch(Circle((0, 0), r_out, fill=False, ec=cor, lw=lw, alpha=alpha))
        for j in range(n):
            a = np.pi / 2 + 2 * np.pi * j / n
            ax.plot([r_in * np.cos(a), r_out * np.cos(a)],
                    [r_in * np.sin(a), r_out * np.sin(a)], color=cor, lw=lw, alpha=alpha)
        r_in = r_out


def painel_tensoes(ax, fig, volt, args):
    cmap, norm = plt.get_cmap("jet"), Normalize(args.v_min, args.v_max)
    r_in = 0.0
    for r_out, n, primeiro in zip(R_ANEIS, N_SETORES, PRIMEIRO_SEG):
        for j in range(n):
            t1 = 90 + 360 * j / n
            t2 = t1 + 360 / n
            k = primeiro + j
            ax.add_patch(Wedge((0, 0), r_out, t1, t2, width=r_out - r_in,
                               fc=cmap(norm(volt[k - 1])), ec="k", lw=0.6))
            a = np.deg2rad((t1 + t2) / 2)
            rm = (r_in + r_out) / 2
            ax.text(rm * np.cos(a), rm * np.sin(a), str(k), ha="center", va="center",
                    fontsize=6.5, color="white",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
        r_in = r_out
    ax.add_patch(Circle((0, 0), args.pupila_mm / 2, fill=False, ec="white", ls="--", lw=0.9))
    ax.set_xlim(-R_ESPELHO * 1.03, R_ESPELHO * 1.03)
    ax.set_ylim(-R_ESPELHO * 1.03, R_ESPELHO * 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Tensões nos segmentos (pupila tracejada)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    barra_cores(fig, sm, ax, "Tensão (V)")


def painel_mapa(ax, fig, x, dados, mascara, titulo, rotulo, cmap, args, simetrico=True):
    ext = [x[0] * 1e3, x[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3]
    mapa = np.ma.masked_where(~mascara, dados)
    vmax = np.abs(mapa).max() or 1.0
    im = ax.imshow(mapa, extent=ext, origin="lower", cmap=cmap,
                   vmin=-vmax if simetrico else None, vmax=vmax if simetrico else None)
    contornos_segmentos(ax, cor="k", lw=0.5, alpha=0.5)
    ax.set_xlim(-R_ESPELHO * 1.03, R_ESPELHO * 1.03)
    ax.set_ylim(-R_ESPELHO * 1.03, R_ESPELHO * 1.03)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(titulo)
    barra_cores(fig, im, ax, rotulo)


def painel_foco(ax, fig, xf, I, titulo):
    ext = [xf[0] * 1e6, xf[-1] * 1e6, xf[0] * 1e6, xf[-1] * 1e6]
    im = ax.imshow(I, extent=ext, origin="lower", cmap=CMAP_LASER, vmin=0, vmax=1)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(titulo)
    barra_cores(fig, im, ax, "Intensidade (I/I$_{0,\\,ref}$)")


def figura(volt, res, args, nome):
    fig, axs = plt.subplots(2, 3, figsize=(17, 10), facecolor="#101010",
                            gridspec_kw={"hspace": 0.35, "wspace": 0.38})
    x, xf = res["x"], res["xf"]
    mirror = res["ids"] > 0

    painel_tensoes(axs[0, 0], fig, volt, args)
    painel_mapa(axs[0, 1], fig, x, res["z"] * 1e9, mirror,
                "Deformação da superfície do espelho", "z (nm)", "RdBu_r", args)
    painel_mapa(axs[0, 2], fig, x, res["fase"], res["pupila"],
                "Fase imposta ao feixe na pupila", "Fase (rad)", "twilight_shifted", args)
    painel_foco(axs[1, 0], fig, xf, res["I_ref"], "Plano focal: espelho plano (referência)")
    painel_foco(axs[1, 1], fig, xf, res["I_def"], "Plano focal: espelho deformado")

    # cortes 1D passando pelo pico
    ax = axs[1, 2]
    j, i = np.unravel_index(np.argmax(res["I_def"]), res["I_def"].shape)
    c = len(xf) // 2
    ax.plot(xf * 1e6, res["I_ref"][c, :], "w--", lw=1.2, label="Referência (corte em x)")
    ax.plot(xf * 1e6, res["I_def"][j, :], color="#ffa500", lw=1.6, label="Deformado (corte em x)")
    ax.plot(xf * 1e6, res["I_def"][:, i], color="#00bfff", lw=1.6, label="Deformado (corte em y)")
    ax.set_xlabel("Posição (µm)")
    ax.set_ylabel("Intensidade (I/I$_{0,\\,ref}$)")
    ax.set_title("Cortes do perfil de intensidade no foco")
    ax.grid(alpha=0.25)
    ax.legend(facecolor="#202020", labelcolor=TXT, edgecolor="#555", fontsize=8)

    for a in axs.ravel():
        estilo_escuro(a)

    fig.suptitle(
        f"{nome} — modelo {args.modelo} | $\\lambda$ = {args.lambda_nm:.0f} nm, "
        f"w = {args.w_mm:.1f} mm, f = {args.foco_mm:.0f} mm | "
        f"RMS da frente de onda = {res['rms_nm']:.0f} nm | "
        f"razão de pico (Strehl) = {res['strehl']:.3f}",
        color=TXT, fontsize=13)
    return fig

def plot_fase_3d(res, args, nome):
    """Superfície 3D da fase (em radianos) da pupila do DM"""
    x_mm = res["x"] * 1e3
    R = args.pupila_mm / 2
    
    # recortando a região da pupila e a mascara fora dela
    sel = np.abs(x_mm) <= R * 1.01
    xs = x_mm[sel]
    fase = np.where(res["pupila"], res["fase"], np.nan)[np.ix_(sel, sel)]
    X, Y = np.meshgrid(xs, xs)
    
    vmax = np.nanmax(np.abs(fase)) or 1.0
    
    fig = plt.figure(figsize=(11, 8), facecolor="#101010")
    ax = fig.add_subplot(111, projection="3d", facecolor="#101010")
    surf = ax.plot_surface(X, Y, fase, cmap="turbo", vmin = -vmax, vmax = vmax, rstride = 3, cstride=3, linewidth=0, antialiased=True)
    
    # projeção em contornos no "chão" do gráfico
    ax.contourf(X, Y, fase, zdir="z", offset=-vmax * 1.6, levels=20, cmap="turbo", vmin=-vmax, vmax=vmax, alpha=0.8)
    
    ax.set_zlim(-vmax * 1.6, vmax * 1.2)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=28, azim=-55)
    
    ax.set_xlabel("x (mm)", color="white")
    ax.set_ylabel("y (mm)", color="white")
    ax.set_zlabel("Fase (rad)", color="white")
    ax.tick_params(colors="white")
    
    for eixo in (ax.xaxis, ax.yaxis, ax.zaxis):
        eixo.set_pane_color((0.1, 0.1, 0.1, 1.0))
    
    cb = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="Fase (rad)")
    cb.ax.yaxis.label.set_color("white")
    cb.ax.tick_params(colors="white")
    
    ax.set_title(f"Fase imposta pelo DM - {nome}\n"
                 f"$\\lambda$ = {args.lambda_nm:.0f} nm | pico-vale = "
                 f"{np.nanmax(fase) - np.nanmin(fase):.2f} rad | RMS da frente de onda = "
                f"{res['rms_nm']:.0f} nm",
                color=TXT, fontsize=12)
    return fig    
    
def salvar_foco(res, args, nome_arquivo_saida, dpi=200):
    """Salva uma imagem individual contendo apenas o mapa de intensidade 2D
    do feixe no plano focal para o espelho deformado."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#101010")
    
    xf = res["xf"]
    ext = [xf[0] * 1e6, xf[-1] * 1e6, xf[0] * 1e6, xf[-1] * 1e6]
    
    im = ax.imshow(res["I_def"], extent=ext, origin="lower", cmap=CMAP_LASER, vmin=0, vmax=1)
    
    ax.set_xlabel("x (µm)", color=TXT)
    ax.set_ylabel("y (µm)", color=TXT)
    ax.set_title("Plano Focal (Espelho Deformado)", color=TXT, fontsize=11)
    
    estilo_escuro(ax)
    barra_cores(fig, im, ax, "Intensidade (I/I$_{0,\\,ref}$)")
    
    fig.tight_layout()
    fig.savefig(nome_arquivo_saida, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    
    print(f"Imagem do plano focal salva em: {nome_arquivo_saida}")

# ============================ PROGRAMA ======================================
def main():
    CAMINHO_CSV = "-1.00.csv"

    # Parâmetros: todos vêm dos valores padrão definidos no topo do arquivo
    args = SimpleNamespace(
        modelo=MODELO, lambda_nm=LAMBDA_NM, w_mm=W_MM, pupila_mm=PUPILA_MM,
        foco_mm=FOCO_MM, janela_um=JANELA_UM, v_repouso=V_REPOUSO,
        v_min=V_MIN, v_max=V_MAX, curv=CURV_POR_VOLT,
        coef_nm_v=COEF_NM_POR_V, sigma_mm=SIGMA_MM,
    )

    volt = ler_csv(CAMINHO_CSV)
    res = simular(volt, args)
    fig = figura(volt, res, args, Path(CAMINHO_CSV).name)
    saida = Path(CAMINHO_CSV).with_name(Path(CAMINHO_CSV).stem + "_simulacao.png")
    fig.savefig(saida, dpi=140, facecolor=fig.get_facecolor())
    print(f"RMS = {res['rms_nm']:.1f} nm | Strehl = {res['strehl']:.3f} -> {saida}")
    plt.show()

    # ---- PLot 3d da fase
    
    fig = plot_fase_3d(res, args, Path(CAMINHO_CSV).name)
    saida = Path(CAMINHO_CSV).with_name(Path(CAMINHO_CSV).stem + "_fase3d.png")
    fig.savefig(saida, dpi=150, facecolor=fig.get_facecolor())
    print(f"Figura salva em {saida}")
    plt.show()
    
    saida_foco = Path(CAMINHO_CSV).with_name(Path(CAMINHO_CSV).stem + "_plano_focal.png")
    salvar_foco(res, args, saida_foco)
    

if __name__ == "__main__":
    main()