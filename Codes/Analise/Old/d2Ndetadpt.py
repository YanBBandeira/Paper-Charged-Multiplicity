import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Aplica o estilo ROOT do ALICE/CMS
hep.style.use(hep.style.ROOT)

# =============================================================================
# 1. CONFIGURAÇÕES GLOBAIS E ESTÉTICAS
# =============================================================================
cores = ["crimson", "darkorange", "forestgreen", "royalblue", "purple", "dimgray"]
estilos_linha = ["-", "--", "-.", ":", "-", "--"]

# Mapeamento dos casos e suas respectivas pastas de saída no Pythia
casos = {
    "All Physics": "all",
    "No MPI":      "noMPI",
    "Only MPI":    "onlyMPI",
    "Only ISR":    "onlyISR",
    "Only FSR":    "onlyFSR",
    "No Extra":    "noExtra",
}

particulas_info = {
    "Charged": {
        "sigla": "ch", 
        "label": r"Charged Particles ($h^{\pm}$)",
        "xlabel_pt": r"$p_T \ (\mathrm{GeV}/c)$",
        "xlabel_eta": r"$\eta$",
    },
    "Kaons": {
        "sigla": "k",
        "label": r"Kaons",
        "xlabel_pt": r"$p_T \ (\mathrm{GeV}/c)$",
        "xlabel_eta": r"$\eta$",
    },
    "D_mesons": {
        "sigla": "d",
        "label": r"$D$ Mesons",
        "xlabel_pt": r"$p_T \ (\mathrm{GeV}/c)$",
        "xlabel_eta": r"$\eta$",
    },
    "Ds_mesons": {
        "sigla": "ds",
        "label": r"$D_s$ Mesons",
        "xlabel_pt": r"$p_T \ (\mathrm{GeV}/c)$",
        "xlabel_eta": r"$\eta$",
    },
}

# Listas de cortes em Eta e pT
cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
cortes_pt = [2.0, 5.0, 10.0, 20.0, 50.0]  # Ajuste conforme os cortes gerados no seu C++

# Arquivo de log para validação
log_file = open("validacao_simulacao_espectros.log", "w", encoding="utf-8")

def log_print(texto, file=log_file):
    print(texto)
    file.write(texto + "\n")

# =============================================================================
# 2. FUNÇÃO DE LEITURA E VALIDAÇÃO DOS HISTOGRAMAS DO PYTHIA
# =============================================================================
def processar_e_validar_distribuicao(arq_dados):
    """
    Lê os arquivos .dat gerados pelo método .table() do Pythia8.
    """
    if not os.path.exists(arq_dados):
        return None, "Arquivo ausente"

    try:
        dados = np.loadtxt(arq_dados)
    except Exception:
        return None, "Erro ao ler"

    if dados.ndim == 1:
        dados = dados.reshape(1, -1)
    
    if dados.shape[0] == 0 or dados.shape[1] < 2:
        return None, "Arquivo vazio"

    eixo_x = dados[:, 0]
    eixo_y = dados[:, 1]
    
    if np.any(eixo_y < 0):
        return None, "Valores negativos!"

    largura_bin = eixo_x[1] - eixo_x[0] if len(eixo_x) > 1 else 1.0
    integral = np.sum(eixo_y * largura_bin)
    status = "OK" if integral >= 0 else "Erro Integral"

    resultado = {
        "X": eixo_x,
        "Y": eixo_y,
        "integral": integral,
        "bin_width": largura_bin
    }
    return resultado, status

# =============================================================================
# 3. FUNÇÃO DE SALVAR GRÁFICO
# =============================================================================
def salvar_grafico(dados_plot, xlabel, ylabel, label, nome_pdf, y_log=True, leg_title="", subpasta=""):
    caminho_dir = os.path.join("Figures", subpasta)
    os.makedirs(caminho_dir, exist_ok=True)
    caminho_completo = os.path.join(caminho_dir, nome_pdf)

    plt.figure(figsize=(9, 6))

    max_x_global = 0
    max_y_global = 0
    min_y_global = float('inf')

    for item in dados_plot:
        x_vals = item["x"]
        y_vals = np.array(item["y"], dtype=float)
        
        if len(x_vals) == 0: continue
        
        y_vals_plot = np.where(y_vals <= 0, 1e-12, y_vals) if y_log else y_vals
        largura_bin = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 1.0
        bin_edges = np.append(x_vals - largura_bin / 2.0, x_vals[-1] + largura_bin / 2.0)

        hep.histplot(
            y_vals_plot,
            bins=bin_edges,
            histtype='step',
            label=item["label"],
            color=item["color"],
            linestyle=item.get("ls", "-"),
            linewidth=1.8
        )

        max_x_global = max(max_x_global, np.max(x_vals))
        max_y_global = max(max_y_global, np.max(y_vals))
        valid_y = y_vals[y_vals > 0]
        if len(valid_y) > 0:
            min_y_global = min(min_y_global, np.min(valid_y))

    if y_log:
        plt.yscale("log")
        bottom_lim = min_y_global * 0.1 if min_y_global != float('inf') else 1e-6
        top_lim = max_y_global * 50 if max_y_global > 0 else 1.0
        plt.ylim(bottom=bottom_lim, top=top_lim)
    else:
        top_lim = max_y_global * 1.35 if max_y_global > 0 else 1.0
        plt.ylim(bottom=0, top=top_lim)

    plt.xlim(left=0 if "eta" not in xlabel.lower() else -max_x_global, right=max_x_global * 1.02)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(0.4, 0.9, label, transform=plt.gca().transAxes, fontsize=12, verticalalignment="center")
    plt.grid(True, which="both", linestyle="--", alpha=0.2)
    plt.legend(loc="upper right", fontsize=10, title=leg_title, title_fontsize=11, frameon=False)

    plt.tight_layout()
    plt.savefig(caminho_completo, bbox_inches="tight")
    plt.close()

# =============================================================================
# 4. LOOP PRINCIPAL DE ANÁLISE E PLOTAGEM
# =============================================================================
log_print(f"{'='*85}\n INICIANDO ANÁLISE DOS ESPECTROS (Pythia Output)\n{'='*85}")

for regime in ["Soft", "Hard"]:
    log_print(f"\n---> REGIME: {regime}")

    # -------------------------------------------------------------------------
    # A) ANÁLISE DE dN/deta COM CORTE EM pT (ex: _dndeta_ch_pt_lt_10GeV.dat)
    # -------------------------------------------------------------------------
    for pt_corte in cortes_pt:
        pt_str = f"{pt_corte:.0f}"
        log_print(f"\n   • Corte em pT: pT < {pt_corte} GeV")

        for p_nome, p_conf in particulas_info.items():
            sigla = p_conf["sigla"]
            dados_dndeta_pt = []

            for idx, (nome_caso, subpasta_sufixo) in enumerate(casos.items()):
                cor, estilo = cores[idx % len(cores)], estilos_linha[idx % len(estilos_linha)]
                
                if os.path.exists(os.path.join("Dados", regime, subpasta_sufixo)):
                    dir_dados = os.path.join("Dados", regime, subpasta_sufixo)
                else:
                    dir_dados = os.path.join("..", "Dados", regime, subpasta_sufixo)

                nome_script = f"{regime}_14TeV_{subpasta_sufixo}"
                arq_dados = os.path.join(dir_dados, f"{nome_script}_dndeta_{sigla}_pt_lt_{pt_str}GeV.dat")

                res, status = processar_e_validar_distribuicao(arq_dados)
                if res:
                    dados_dndeta_pt.append({"x": res["X"], "y": res["Y"], "label": nome_caso, "color": cor, "ls": estilo})

            if dados_dndeta_pt:
                sub_dir = os.path.join(f"{regime}_QCD", p_nome, "dN_deta_pt_cuts")
                salvar_grafico(dados_dndeta_pt, p_conf["xlabel_eta"], r"$dN/d\eta$", p_conf["label"], 
                               f"{p_nome}_dndeta_pt_lt_{pt_str}GeV.pdf", y_log=False, 
                               leg_title=rf"$p_T < {pt_str}\ \mathrm{{GeV/c}}$ ({regime})", subpasta=sub_dir)

    # -------------------------------------------------------------------------
    # B) ESPECTROS DE pT COM CORTE EM ETA (ex: _pt_ch_eta_0p5.dat)
    # -------------------------------------------------------------------------
    for corte_eta in cortes_eta:
        corte_str = f"{corte_eta:.1f}".replace(".", "p")
        log_print(f"\n   • Espectros de pT com corte em |eta| < {corte_eta}")

        for p_nome, p_conf in particulas_info.items():
            sigla = p_conf["sigla"]
            dados_pt_caso = []

            for idx, (nome_caso, subpasta_sufixo) in enumerate(casos.items()):
                cor, estilo = cores[idx % len(cores)], estilos_linha[idx % len(estilos_linha)]
                
                if os.path.exists(os.path.join("Dados", regime, subpasta_sufixo)):
                    dir_dados = os.path.join("Dados", regime, subpasta_sufixo)
                else:
                    dir_dados = os.path.join("..", "Dados", regime, subpasta_sufixo)

                nome_script = f"{regime}_14TeV_{subpasta_sufixo}"
                arq_dados = os.path.join(dir_dados, f"{nome_script}_pt_{sigla}_eta_{corte_str}.dat")

                res, status = processar_e_validar_distribuicao(arq_dados)
                if res:
                    dados_pt_caso.append({"x": res["X"], "y": res["Y"], "label": nome_caso, "color": cor, "ls": estilo})

            if dados_pt_caso:
                sub_dir = os.path.join(f"{regime}_QCD", p_nome, "Spectra_pT")
                salvar_grafico(dados_pt_caso, p_conf["xlabel_pt"], r"Entries / Bin", p_conf["label"], 
                               f"{p_nome}_pt_eta_{corte_str}.pdf", y_log=True, 
                               leg_title=rf"$|\eta| < {corte_eta}$ ({regime})", subpasta=sub_dir)

msg_fim = f"\n{'='*85}\nProcessamento e plotagem concluídos com sucesso!\nLog salvo em: 'validacao_simulacao_espectros.log'\n{'='*85}"
log_print(msg_fim)
log_file.close()