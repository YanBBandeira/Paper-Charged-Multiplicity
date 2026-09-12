import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Aplica o estilo clássico do ALICE/ROOT
hep.style.use(hep.style.ROOT)

# =============================================================================
# 1. CONFIGURAÇÕES GLOBAIS E ESTÉTICAS
# =============================================================================
cores = ["crimson", "darkorange", "forestgreen", "royalblue", "purple", "dimgray"]
estilos_linha = ["-", "--", "-.", ":", "-", "--"]

casos = {
    "All Physics": ("Soft_14TeV_all", "Hard_14TeV_all"),
    "No MPI":      ("Soft_14TeV_noMPI", "Hard_14TeV_noMPI"),
    "Only MPI":    ("Soft_14TeV_onlyMPI", "Hard_14TeV_onlyMPI"),
    "Only ISR":    ("Soft_14TeV_onlyISR", "Hard_14TeV_onlyISR"),
    "Only FSR":    ("Soft_14TeV_onlyFSR", "Hard_14TeV_onlyFSR"),
    "No Extra":    ("Soft_14TeV_noExtra", "Hard_14TeV_noExtra"),
}

particulas_info = {
    "Charged": {
        "sigla": "ch", 
        "label": r"Charged Particles ($h^{\pm}$)",
        "xlabel_eta": r"$\eta$",
        "xlabel_pt": r"$p_T$ (GeV/$c$)",
    },
    "Kaons": {
        "sigla": "k",
        "label": r"Kaons",
        "xlabel_eta": r"$\eta$",
        "xlabel_pt": r"$p_T$ (GeV/$c$)",
    },
    "D_mesons": {
        "sigla": "d",
        "label": r"$D$ Mesons",
        "xlabel_eta": r"$\eta$",
        "xlabel_pt": r"$p_T$ (GeV/$c$)",
    },
    "Ds_mesons": {
        "sigla": "ds",
        "label": r"$D_s$ Mesons",
        "xlabel_eta": r"$\eta$",
        "xlabel_pt": r"$p_T$ (GeV/$c$)",
    },
}

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
cortes_pt_dndeta = [5.0, 10.0, 20.0, 40.0]

# Abre o arquivo de log no modo de escrita na raiz para auditoria das distribuições 1D
log_file = open("validacao_simulacao_1d.log", "w", encoding="utf-8")

def log_print(texto, file=log_file):
    """Escreve no console e no arquivo de log simultaneamente."""
    print(texto)
    file.write(texto + "\n")

# =============================================================================
# 2. FUNÇÃO DE LEITURA, VALIDAÇÃO CRUZADA E INTEGRAÇÃO 1D
# =============================================================================
def processar_e_validar_1d(arq_1d, arq_mult, tipo="eta"):
    """
    Lê o arquivo 1D (.dat), calcula sua integral de área e a compara 
    com a média esperada da multiplicidade para atestar a normalização.
    """
    if not os.path.exists(arq_1d) or not os.path.exists(arq_mult):
        return None, "Arquivos ausentes"

    # Leitura do espectro 1D
    dados_1d = np.loadtxt(arq_1d)
    if dados_1d.ndim == 1:
        dados_1d = dados_1d.reshape(1, -1)
        
    x_vals = dados_1d[:, 0]
    y_vals = dados_1d[:, 1]
    
    if len(x_vals) < 2:
        return None, "Bins insuficientes"

    dx = x_vals[1] - x_vals[0]
    integral_1d = np.sum(y_vals * dx)

    # Leitura do arquivo de multiplicidade para obter a referência de <N>
    dados_mult = np.loadtxt(arq_mult)
    if dados_mult.ndim == 1:
        dados_mult = dados_mult.reshape(1, -1)
        
    N_valores = dados_mult[:, 0]
    counts = dados_mult[:, 1]
    total_eventos = np.sum(counts)
    
    if total_eventos == 0:
        return None, "Zero eventos"
        
    media_N_esperada = np.sum(N_valores * (counts / total_eventos))

    # Teste de Sanidade (Para dN/deta, a integral em eta dá a multiplicidade total na faixa. 
    # Para dN/dpt, a integral dá a fração do momento, logo testamos principalmente o dN/deta).
    status = "OK"
    if tipo == "eta":
        if not np.isclose(integral_1d, media_N_esperada, rtol=1e-2, atol=1e-2):
            status = "Erro Integral dN/deta"

    resultado = {
        "x": x_vals,
        "y": y_vals,
        "integral": integral_1d,
        "media_N": media_N_esperada
    }
    return resultado, status

# =============================================================================
# 3. FUNÇÃO DE SALVAR GRÁFICO 1D
# =============================================================================
def salvar_grafico_1d(dados_plot, xlabel, ylabel, label, nome_pdf, y_log=True, leg_title="", subpasta=""):
    caminho_dir = os.path.join("Figures", subpasta)
    os.makedirs(caminho_dir, exist_ok=True)
    caminho_completo = os.path.join(caminho_dir, nome_pdf)

    plt.figure(figsize=(9, 6))
    max_y_global = 0

    for item in dados_plot:
        x_vals = item["x"]
        y_vals = np.array(item["y"], dtype=float)
        
        if len(x_vals) == 0: continue
        
        largura_bin = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 1.0
        bin_edges = np.append(x_vals - largura_bin / 2.0, x_vals[-1] + largura_bin / 2.0)

        hep.histplot(
            y_vals,
            bins=bin_edges,
            histtype='step',
            label=item["label"],
            color=item["color"],
            linestyle=item.get("ls", "-"),
            linewidth=1.8
        )

        local_max_y = np.max(y_vals)
        if local_max_y > max_y_global:
            max_y_global = local_max_y

    if y_log:
        plt.yscale("log")
        plt.ylim(bottom=1e-4, top=max_y_global * 100 if max_y_global > 0 else 1.0)
    else:
        plt.ylim(bottom=0, top=max_y_global * 1.35 if max_y_global > 0 else 1.0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(0.4, 0.9, label, transform=plt.gca().transAxes, fontsize=12, verticalalignment="center")
    plt.grid(True, which="both", linestyle="--", alpha=0.2)
    plt.legend(loc="upper right", fontsize=10, title=leg_title, title_fontsize=11, frameon=False)

    plt.tight_layout()
    plt.savefig(caminho_completo, bbox_inches="tight")
    plt.close()

# =============================================================================
# 4. PROCESSAMENTO DOS ESPECTROS 1D POR CORTE EM ETA (COM LOG)
# =============================================================================
for corte_eta in cortes_eta:
    corte_str = f"{corte_eta:.1f}".replace(".", "p")
    
    msg_separador = f"\n{'='*85}\n PROCESSANDO E VALIDANDO 1D EM |eta| < {corte_eta}\n{'='*85}"
    log_print(msg_separador)

    for p_nome, p_conf in particulas_info.items():
        sigla = p_conf["sigla"]
        
        dados_eta_soft, dados_pt_soft = [], []
        dados_eta_hard, dados_pt_hard = [], []

        cabecalho = f"\n--- Partícula: {p_nome} (dN/deta) ---\n" + \
                    f"{'Configuração':<15} | {'Processo':<6} | {'Integral dN/deta':<15} | {'<N> Esperado':<12} | {'Status'}\n" + \
                    "-" * 62
        log_print(cabecalho)

        for idx, (nome_caso, (pref_soft, pref_hard)) in enumerate(casos.items()):
            pasta_soft = pref_soft.split("_")[-1]
            pasta_hard = pref_hard.split("_")[-1]
            
            cor = cores[idx % len(cores)]
            estilo = estilos_linha[idx % len(estilos_linha)]

            # -----------------------------------------------------------------
            # 4.1. Soft: dN/deta e pT
            # -----------------------------------------------------------------
            arq_eta_soft  = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_dndeta_{sigla}_eta_{corte_str}.dat")
            arq_pt_soft   = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_pt_{sigla}_eta_{corte_str}.dat")
            arq_mult_soft = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_mult_{sigla}_eta_{corte_str}.dat")
            
            res_eta_s, status_eta_s = processar_e_validar_1d(arq_eta_soft, arq_mult_soft, tipo="eta")
            if res_eta_s is not None:
                linha_s = f"{nome_caso:<15} | {'Soft':<6} | {res_eta_s['integral']:<15.4f} | {res_eta_s['media_N']:<12.4f} | {status_eta_s}"
                log_print(linha_s)
                dados_eta_soft.append({"x": res_eta_s["x"], "y": res_eta_s["y"], "label": nome_caso, "color": cor, "ls": estilo})

            if os.path.exists(arq_pt_soft):
                d = np.loadtxt(arq_pt_soft)
                if d.ndim == 1: d = d.reshape(1, -1)
                dados_pt_soft.append({"x": d[:, 0], "y": d[:, 1], "label": nome_caso, "color": cor, "ls": estilo})

            # -----------------------------------------------------------------
            # 4.2. Hard: dN/deta e pT
            # -----------------------------------------------------------------
            arq_eta_hard  = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_dndeta_{sigla}_eta_{corte_str}.dat")
            arq_pt_hard   = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_pt_{sigla}_eta_{corte_str}.dat")
            arq_mult_hard = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_mult_{sigla}_eta_{corte_str}.dat")
            
            res_eta_h, status_eta_h = processar_e_validar_1d(arq_eta_hard, arq_mult_hard, tipo="eta")
            if res_eta_h is not None:
                linha_h = f"{nome_caso:<15} | {'Hard':<6} | {res_eta_h['integral']:<15.4f} | {res_eta_h['media_N']:<12.4f} | {status_eta_h}"
                log_print(linha_h)
                dados_eta_hard.append({"x": res_eta_h["x"], "y": res_eta_h["y"], "label": nome_caso, "color": cor, "ls": estilo})

            if os.path.exists(arq_pt_hard):
                d = np.loadtxt(arq_pt_hard)
                if d.ndim == 1: d = d.reshape(1, -1)
                dados_pt_hard.append({"x": d[:, 0], "y": d[:, 1], "label": nome_caso, "color": cor, "ls": estilo})

        sub_dir_soft = os.path.join("Soft_QCD", p_nome)
        sub_dir_hard = os.path.join("Hard_QCD", p_nome)

        # Salvamento dN/deta vs eta
        if dados_eta_soft:
            salvar_grafico_1d(dados_eta_soft, p_conf["xlabel_eta"], r"$\mathrm{d}N/\mathrm{d}\eta$", p_conf["label"], 
                              f"{p_nome}_dNdeta_eta_{corte_str}.pdf", y_log=False, leg_title=rf"$|\eta| < {corte_eta}$ (Soft)", subpasta=sub_dir_soft)
        if dados_eta_hard:
            salvar_grafico_1d(dados_eta_hard, p_conf["xlabel_eta"], r"$\mathrm{d}N/\mathrm{d}\eta$", p_conf["label"], 
                              f"{p_nome}_dNdeta_eta_{corte_str}.pdf", y_log=False, leg_title=rf"$|\eta| < {corte_eta}$ (Hard)", subpasta=sub_dir_hard)

        # Salvamento dN/dpt vs pt
        if dados_pt_soft:
            salvar_grafico_1d(dados_pt_soft, p_conf["xlabel_pt"], r"$\mathrm{d}N/\mathrm{d}p_T$", p_conf["label"], 
                              f"{p_nome}_dNdpt_eta_{corte_str}.pdf", y_log=True, leg_title=rf"$|\eta| < {corte_eta}$ (Soft)", subpasta=sub_dir_soft)
        if dados_pt_hard:
            salvar_grafico_1d(dados_pt_hard, p_conf["xlabel_pt"], r"$\mathrm{d}N/\mathrm{d}p_T$", p_conf["label"], 
                              f"{p_nome}_dNdpt_eta_{corte_str}.pdf", y_log=True, leg_title=rf"$|\eta| < {corte_eta}$ (Hard)", subpasta=sub_dir_hard)

# =============================================================================
# 5. PROCESSAMENTO DOS ESPECTROS dN/deta COM CORTE SUPERIOR EM PT
# =============================================================================
for pt_corte in cortes_pt_dndeta:
    pt_str = f"{pt_corte:.0f}"
    
    print(f"\nProcessando dN/deta com corte pt < {pt_corte} GeV")

    for p_nome, p_conf in particulas_info.items():
        sigla = p_conf["sigla"]
        dados_pt_cut_soft, dados_pt_cut_hard = [], []

        for idx, (nome_caso, (pref_soft, pref_hard)) in enumerate(casos.items()):
            pasta_soft = pref_soft.split("_")[-1]
            pasta_hard = pref_hard.split("_")[-1]
            
            cor = cores[idx % len(cores)]
            estilo = estilos_linha[idx % len(estilos_linha)]

            arq_s = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_dndeta_{sigla}_pt_lt_{pt_str}GeV.dat")
            if os.path.exists(arq_s):
                d = np.loadtxt(arq_s)
                if d.ndim == 1: d = d.reshape(1, -1)
                dados_pt_cut_soft.append({"x": d[:, 0], "y": d[:, 1], "label": nome_caso, "color": cor, "ls": estilo})

            arq_h = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_dndeta_{sigla}_pt_lt_{pt_str}GeV.dat")
            if os.path.exists(arq_h):
                d = np.loadtxt(arq_h)
                if d.ndim == 1: d = d.reshape(1, -1)
                dados_pt_cut_hard.append({"x": d[:, 0], "y": d[:, 1], "label": nome_caso, "color": cor, "ls": estilo})

        sub_dir_soft = os.path.join("Soft_QCD", p_nome)
        sub_dir_hard = os.path.join("Hard_QCD", p_nome)

        if dados_pt_cut_soft:
            salvar_grafico_1d(dados_pt_cut_soft, p_conf["xlabel_eta"], r"$\mathrm{d}N/\mathrm{d}\eta$", p_conf["label"], 
                              f"{p_nome}_dndeta_pt_lt_{pt_str}GeV.pdf", y_log=False, leg_title=rf"$p_T < {pt_corte}\ \mathrm{{GeV/c}}$ (Soft)", subpasta=sub_dir_soft)
        if dados_pt_cut_hard:
            salvar_grafico_1d(dados_pt_cut_hard, p_conf["xlabel_eta"], r"$\mathrm{d}N/\mathrm{d}\eta$", p_conf["label"], 
                              f"{p_nome}_dndeta_pt_lt_{pt_str}GeV.pdf", y_log=False, leg_title=rf"$p_T < {pt_corte}\ \mathrm{{GeV/c}}$ (Hard)", subpasta=sub_dir_hard)

msg_fim = f"\n{'='*85}\nProcessamento, testes de validação e plots 1D concluídos!\nLog salvo em 'validacao_simulacao_1d.log'\n{'='*85}"
log_print(msg_fim)
log_file.close()