import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


hep.style.use(hep.style.ROOT)

# =============================================================================
# 1. CONFIGURACOES
# =============================================================================
casos = {
    "All Physics": ("Soft_14TeV_all", "Hard_14TeV_all"),
    "No MPI": ("Soft_14TeV_noMPI", "Hard_14TeV_noMPI"),
    "Only MPI": ("Soft_14TeV_onlyMPI", "Hard_14TeV_onlyMPI"),
    "Only ISR": ("Soft_14TeV_onlyISR", "Hard_14TeV_onlyISR"),
    "Only FSR": ("Soft_14TeV_onlyFSR", "Hard_14TeV_onlyFSR"),
    "No Extra": ("Soft_14TeV_noExtra", "Hard_14TeV_noExtra"),
}

particulas_info = {
    "Charged": {"sigla": "ch", "label": r"Charged Particles ($h^{\pm}$)"},
    "D0_mesons": {"sigla": "d0", "label": r"$D^0$ Mesons"},
    "D_mesons": {"sigla": "d", "label": r"$D^\pm$ Mesons"},
    "Ds_mesons": {"sigla": "ds", "label": r"$D_s$ Mesons"},
}

cores = ["crimson", "darkorange", "forestgreen", "royalblue", "purple", "dimgray"]
estilos_linha = ["-", "--", "-.", ":", "-", "--"]
cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

log_file = open("validacao_simulacao_dndpt.log", "w", encoding="utf-8")


def log_print(texto):
    """Escreve simultaneamente no terminal e no arquivo de validacao."""
    print(texto)
    log_file.write(texto + "\n")


def carregar_espectro_pt(arq_pt, arq_mult, nome_caso, nome_particula, corte_eta):
    """Le o histograma 1D h_pt e normaliza dN/dpT por evento."""
    identificacao = f"{nome_caso} | {nome_particula} | eta_{corte_eta}"

    if not os.path.exists(arq_pt):
        log_print(f"  [AVISO | {identificacao}]: Histograma 1D de pT ausente")
        return None
    if not os.path.exists(arq_mult):
        log_print(f"  [AVISO | {identificacao}]: Histograma de multiplicidade ausente")
        return None

    # Os arquivos exportados por Hist nao possuem cabecalho: a primeira linha
    # ja e o bin fisico centrado em pT = 0.05 GeV/c.
    dados_pt = np.loadtxt(arq_pt)
    if dados_pt.ndim == 1:
        dados_pt = dados_pt.reshape(1, -1)
    if dados_pt.shape[1] < 2:
        log_print(f"  [AVISO | {identificacao}]: Formato invalido no histograma de pT")
        return None

    pt_vals = dados_pt[:, 0]
    conteudo_bruto = dados_pt[:, 1]
    if len(pt_vals) < 2:
        log_print(f"  [AVISO | {identificacao}]: Bins insuficientes no histograma de pT")
        return None

    largura_pt = np.median(np.diff(pt_vals))
    if largura_pt <= 0:
        log_print(f"  [AVISO | {identificacao}]: Largura de bin de pT invalida")
        return None

    dados_mult = np.loadtxt(arq_mult)
    if dados_mult.ndim == 1:
        dados_mult = dados_mult.reshape(1, -1)
    n_eventos = np.sum(dados_mult[:, 1])
    if n_eventos <= 0:
        log_print(f"  [AVISO | {identificacao}]: Numero de eventos invalido")
        return None

    dndpt_normalizado = conteudo_bruto / (n_eventos * largura_pt)
    integral_dndpt = np.sum(dndpt_normalizado) * largura_pt

    # N=0 e um bin fisico e deve permanecer no calculo da media.
    total_counts = np.sum(dados_mult[:, 1])
    media_mult = np.sum(dados_mult[:, 0] * dados_mult[:, 1]) / total_counts
    diferenca = abs(integral_dndpt - media_mult) / media_mult * 100 if media_mult > 0 else 0

    log_print(
        f"  [Sanidade OK | {identificacao}]: N_ev = {int(n_eventos)} | "
        f"Integral dN/dpT = {integral_dndpt:.4f} | "
        f"Media <N> (Mult) = {media_mult:.4f} (Dif: {diferenca:.2f}%)"
    )
    return {"pt": pt_vals, "dndpt": dndpt_normalizado}


def salvar_grafico_dndpt(dados_plot, label_particula, nome_pdf, leg_title, subpasta):
    """Salva a comparacao dos seis casos para um corte de eta."""
    caminho_dir = os.path.join("Figures", subpasta)
    os.makedirs(caminho_dir, exist_ok=True)
    caminho_completo = os.path.join(caminho_dir, nome_pdf)

    plt.figure(figsize=(9, 6))
    max_y = 0.0
    max_pt_nonzero = 0.0

    for item in dados_plot:
        pt_vals = item["pt"]
        dndpt_vals = np.asarray(item["dndpt"], dtype=float)
        mask = (pt_vals >= 0) & (dndpt_vals > 0)
        if not np.any(mask):
            continue

        if len(pt_vals) > 1:
            largura_pt = np.median(np.diff(pt_vals))
        else:
            largura_pt = 0.1
        bin_edges = np.append(pt_vals - largura_pt / 2, pt_vals[-1] + largura_pt / 2)

        hep.histplot(
            dndpt_vals,
            bins=bin_edges,
            histtype="step",
            label=item["label"],
            color=item["color"],
            linestyle=item["ls"],
            linewidth=1.8,
        )
        max_y = max(max_y, np.max(dndpt_vals[mask]))
        max_pt_nonzero = max(max_pt_nonzero, np.max(pt_vals[mask]))

    plt.yscale("log")
    plt.xlim(left=0, right=max_pt_nonzero + 5 if max_pt_nonzero > 0 else 1.0)
    if max_y > 0:
        plt.ylim(bottom=max_y * 1e-7, top=max_y * 2)
    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel(r"$1/N_{\mathrm{ev}}\ dN/dp_T$ ((GeV/$c$)$^{-1}$)")
    plt.text(0.05, 0.90, label_particula, transform=plt.gca().transAxes, fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize=9, title=leg_title, title_fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(caminho_completo, bbox_inches="tight")
    plt.close()


# =============================================================================
# 2. PROCESSAMENTO DE TODOS OS CASOS SOFT E HARD
# =============================================================================
for corte_eta in cortes_eta:
    corte_eta_str = f"{corte_eta:.1f}".replace(".", "p")
    tipo_corte_dir = f"eta_{corte_eta_str}/pt_global"
    legenda_corte = rf"$|\eta| < {corte_eta}$"

    log_print(f"\n{'=' * 85}\nPROCESSANDO dN/dpT: |eta| < {corte_eta}\n{'=' * 85}")

    for p_nome, p_conf in particulas_info.items():
        dados_soft = []
        dados_hard = []

        for idx, (nome_caso, (pref_soft, pref_hard)) in enumerate(casos.items()):
            pasta_soft = pref_soft.split("_")[-1]
            pasta_hard = pref_hard.split("_")[-1]
            cor = cores[idx]
            estilo = estilos_linha[idx]

            arq_pt_soft = os.path.join(
                "..", "Dados", "Soft", pasta_soft,
                f"{pref_soft}_pt_{p_conf['sigla']}_eta_{corte_eta_str}.dat",
            )
            arq_mult_soft = os.path.join(
                "..", "Dados", "Soft", pasta_soft,
                f"{pref_soft}_mult_{p_conf['sigla']}_eta_{corte_eta_str}_global.dat",
            )
            resultado_soft = carregar_espectro_pt(
                arq_pt_soft, arq_mult_soft, f"Soft - {nome_caso}", p_nome, corte_eta_str
            )
            if resultado_soft is not None:
                dados_soft.append({**resultado_soft, "label": nome_caso, "color": cor, "ls": estilo})

            arq_pt_hard = os.path.join(
                "..", "Dados", "Hard", pasta_hard,
                f"{pref_hard}_pt_{p_conf['sigla']}_eta_{corte_eta_str}.dat",
            )
            arq_mult_hard = os.path.join(
                "..", "Dados", "Hard", pasta_hard,
                f"{pref_hard}_mult_{p_conf['sigla']}_eta_{corte_eta_str}_global.dat",
            )
            resultado_hard = carregar_espectro_pt(
                arq_pt_hard, arq_mult_hard, f"Hard - {nome_caso}", p_nome, corte_eta_str
            )
            if resultado_hard is not None:
                dados_hard.append({**resultado_hard, "label": nome_caso, "color": cor, "ls": estilo})

        if dados_soft:
            salvar_grafico_dndpt(
                dados_soft,
                p_conf["label"],
                f"{p_nome}_dNdPt.pdf",
                legenda_corte,
                os.path.join("Soft", tipo_corte_dir, p_nome),
            )
        if dados_hard:
            salvar_grafico_dndpt(
                dados_hard,
                p_conf["label"],
                f"{p_nome}_dNdPt.pdf",
                legenda_corte,
                os.path.join("Hard", tipo_corte_dir, p_nome),
            )

log_print(f"\n{'=' * 85}\nProcessamento e geracao dos plots de dN/dpT concluido!\n{'=' * 85}")
log_file.close()
