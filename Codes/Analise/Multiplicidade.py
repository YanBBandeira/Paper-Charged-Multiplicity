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

# Mapeamento das configurações de simulação
casos = {
    "All Physics": ("Soft_14TeV_all", "Hard_14TeV_all"),
    "No MPI":      ("Soft_14TeV_noMPI", "Hard_14TeV_noMPI"),
    "Only MPI":    ("Soft_14TeV_onlyMPI", "Hard_14TeV_onlyMPI"),
    "Only ISR":    ("Soft_14TeV_onlyISR", "Hard_14TeV_onlyISR"),
    "Only FSR":    ("Soft_14TeV_onlyFSR", "Hard_14TeV_onlyFSR"),
    "No Extra":    ("Soft_14TeV_noExtra", "Hard_14TeV_noExtra"),
}

# ATUALIZADO: Inclusão do Ds e separação correta das siglas
particulas_info = {
    "Charged": {
        "sigla": "ch", 
        "label": r"Charged Particles ($h^{\pm}$)",
        "xlabel": r"$N_{\mathrm{ch}}$",
    },
    "Kaons": {
        "sigla": "k",
        "label": r"Kaons",
        "xlabel": r"$N_{K}$",
    },
    "D_mesons": {
        "sigla": "d",
        "label": r"$D$ Mesons",
        "xlabel": r"$N_{D}$",
    },
    "Ds_mesons": {
        "sigla": "ds",
        "label": r"$D_s$ Mesons",
        "xlabel": r"$N_{D_s}$",
    },
}

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
bin_width = 0.1

# Abre o arquivo de log no modo de escrita na raiz
log_file = open("validacao_simulacao_PN.log", "w", encoding="utf-8")

def log_print(texto, file=log_file):
    """Escreve no console e no arquivo de log simultaneamente."""
    print(texto)
    file.write(texto + "\n")

# =============================================================================
# 2. FUNÇÃO DE LEITURA E VALIDAÇÃO CRUZADA (COM TESTES)
# =============================================================================
def processar_e_validar_dados(arq_mult, arq_eta):
    """
    Lê os arquivos .dat, executa os testes de sanidade cruzados 
    e retorna os dados processados e o status da validação.
    """
    if not os.path.exists(arq_mult) or not os.path.exists(arq_eta):
        return None, "Arquivos ausentes"

    # Leitura da Multiplicidade
    dados_mult = np.loadtxt(arq_mult)
    if dados_mult.ndim == 1:
        dados_mult = dados_mult.reshape(1, -1)
    
    N_valores = dados_mult[:, 0]  # Eixo X
    counts = dados_mult[:, 1]     # Eixo Y (Eventos)
    
    total_eventos = np.sum(counts)
    if total_eventos == 0:
        return None, "Zero eventos"
        
    P_N = counts / total_eventos
    media_N = np.sum(N_valores * P_N)
    soma_prob = np.sum(P_N)

    # Leitura do dN/deta (usado apenas para validação estatística da média)
    dados_eta = np.loadtxt(arq_eta)
    if dados_eta.ndim == 1:
        dados_eta = dados_eta.reshape(1, -1)
        
    eta_bins = dados_eta[:, 0]
    soma_particulas = dados_eta[:, 1]
    
    dndeta_dist = soma_particulas / (total_eventos * bin_width)
    integral_dndeta = np.sum(dndeta_dist * bin_width)

    # Status dos Testes
    status = "OK"
    if not np.isclose(soma_prob, 1.0, atol=1e-3):
        status = "Erro P(N)!=1"
    elif not np.isclose(media_N, integral_dndeta, atol=0.2):
        status = "Aviso Overflow/Integral"

    resultado = {
        "N": N_valores,
        "counts": counts,
        "P_N": P_N,
        "media_N": media_N,
        "soma_prob": soma_prob,
        "integral": integral_dndeta
    }
    return resultado, status

# =============================================================================
# 3. FUNÇÃO DE SALVAR GRÁFICO (COM HEADROOM NO EIXO Y)
# =============================================================================
def salvar_grafico(dados_plot, xlabel, ylabel, label, y_min, nome_pdf, y_log=True, leg_title="", subpasta=""):
    """
    Salva os gráficos comparativos utilizando o hep.histplot() do mplhep.
    """
    caminho_dir = os.path.join("Figures", subpasta)
    os.makedirs(caminho_dir, exist_ok=True)
    caminho_completo = os.path.join(caminho_dir, nome_pdf)

    plt.figure(figsize=(9, 6))

    max_x_global = 0
    max_y_global = 0

    for item in dados_plot:
        x_vals = item["x"]
        y_vals = np.array(item["y"], dtype=float)
        
        if len(x_vals) == 0: continue
        
        if y_log:
            y_vals_plot = np.where(y_vals <= 0, 1e-10, y_vals)
        else:
            y_vals_plot = y_vals

        if len(x_vals) > 1:
            largura_bin = x_vals[1] - x_vals[0]
        else:
            largura_bin = 1.0
            
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

        indices_validos = np.where(y_vals > 0)[0]
        if len(indices_validos) > 0:
            max_local_x = x_vals[indices_validos[-1]]
            if max_local_x > max_x_global:
                max_x_global = max_local_x
                
        local_max_y = np.max(y_vals)
        if local_max_y > max_y_global:
            max_y_global = local_max_y

    if y_log:
        plt.yscale("log")
        top_lim = max_y_global * 100 if max_y_global > 0 else 1.0
        plt.ylim(bottom=y_min, top=top_lim)
    else:
        top_lim = max_y_global * 1.35 if max_y_global > 0 else 1.0
        plt.ylim(bottom=0, top=top_lim)

    if max_x_global > 0:
        # Dá um respiro de 15% no eixo X
        plt.xlim(left=0, right=max_x_global * 1.15)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(0.4, 0.9, label, transform=plt.gca().transAxes, fontsize=12, verticalalignment="center")
    plt.grid(True, which="both", linestyle="--", alpha=0.2)
    plt.legend(loc="upper right", fontsize=10, title=leg_title, title_fontsize=11, frameon=False)

    plt.tight_layout()
    plt.savefig(caminho_completo, bbox_inches="tight")
    plt.close()

# =============================================================================
# 4. LOOP PRINCIPAL DE ANÁLISE, VALIDAÇÃO E PLOTAGEM COMPARATIVA
# =============================================================================
for corte_eta in cortes_eta:
    corte_str = f"{corte_eta:.1f}".replace(".", "p")
    
    msg_separador = f"\n{'='*85}\n PROCESSANDO CORTE EM |eta| < {corte_eta}\n{'='*85}"
    log_print(msg_separador)

    for p_nome, p_conf in particulas_info.items():
        sigla = p_conf["sigla"]
        
        dados_pn_soft, dados_evt_soft = [], []
        dados_pn_hard, dados_evt_hard = [], []

        cabecalho = f"\n--- Partícula: {p_nome} ---\n" + \
                    f"{'Configuração':<15} | {'Processo':<6} | {'Soma P(N)':<10} | {'<N>':<8} | {'Integral':<10} | {'Status'}\n" + \
                    "-" * 68
        log_print(cabecalho)

        for idx, (nome_caso, (pref_soft, pref_hard)) in enumerate(casos.items()):
            pasta_soft = pref_soft.split("_")[-1]
            pasta_hard = pref_hard.split("_")[-1]
            
            cor = cores[idx % len(cores)]
            estilo = estilos_linha[idx % len(estilos_linha)]

            # -----------------------------------------------------------------
            # 4.1. Leitura e Validação: Soft
            # -----------------------------------------------------------------
            arq_mult_soft = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_mult_{sigla}_eta_{corte_str}.dat")
            arq_eta_soft  = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_dndeta_{sigla}_eta_{corte_str}.dat")
            res_s, status_s = processar_e_validar_dados(arq_mult_soft, arq_eta_soft)

            if res_s is not None:
                linha_s = f"{nome_caso:<15} | {'Soft':<6} | {res_s['soma_prob']:<10.4f} | {res_s['media_N']:<8.4f} | {res_s['integral']:<10.4f} | {status_s}"
                log_print(linha_s)
                
                dados_pn_soft.append({
                    "x": res_s["N"], 
                    "y": res_s["P_N"], 
                    "label": f"{nome_caso} ($\\langle N \\rangle$ = {res_s['media_N']:.2f})", 
                    "color": cor, 
                    "ls": estilo
                })

                dados_evt_soft.append({
                    "x": res_s["N"], 
                    "y": res_s["counts"], 
                    "label": f"{nome_caso} ($\\langle N \\rangle$ = {res_s['media_N']:.2f})", 
                    "color": cor, 
                    "ls": estilo
                })

            # -----------------------------------------------------------------
            # 4.2. Leitura e Validação: Hard
            # -----------------------------------------------------------------
            arq_mult_hard = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_mult_{sigla}_eta_{corte_str}.dat")
            arq_eta_hard  = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_dndeta_{sigla}_eta_{corte_str}.dat")
            res_h, status_h = processar_e_validar_dados(arq_mult_hard, arq_eta_hard)

            if res_h is not None:
                linha_h = f"{nome_caso:<15} | {'Hard':<6} | {res_h['soma_prob']:<10.4f} | {res_h['media_N']:<8.4f} | {res_h['integral']:<10.4f} | {status_h}"
                log_print(linha_h)

                dados_pn_hard.append({
                    "x": res_h["N"], 
                    "y": res_h["P_N"], 
                    "label": f"{nome_caso} ($\\langle N \\rangle$ = {res_h['media_N']:.2f})", 
                    "color": cor, 
                    "ls": estilo
                })

                dados_evt_hard.append({
                    "x": res_h["N"], 
                    "y": res_h["counts"], 
                    "label": f"{nome_caso} ($\\langle N \\rangle$ = {res_h['media_N']:.2f})", 
                    "color": cor, 
                    "ls": estilo
                })

        # =====================================================================
        # 4.3. GERAÇÃO DOS PLOTS (APENAS P(N) E EVENTOS)
        # =====================================================================
        if dados_pn_soft:
            sub_dir_soft = os.path.join("Soft_QCD", p_nome)
            salvar_grafico(dados_pn_soft, p_conf["xlabel"], r"$P(N)$", p_conf["label"], 5e-7, 
                           f"{p_nome}_PN_eta_{corte_str}.pdf", y_log=True, leg_title=rf"$|\eta| < {corte_eta}$ (Soft)", subpasta=sub_dir_soft)
            salvar_grafico(dados_evt_soft, p_conf["xlabel"], "Number of events", p_conf["label"], 0.5, 
                           f"{p_nome}_Eventos_eta_{corte_str}.pdf", y_log=True, leg_title=rf"$|\eta| < {corte_eta}$ (Soft)", subpasta=sub_dir_soft)

        if dados_pn_hard:
            sub_dir_hard = os.path.join("Hard_QCD", p_nome)
            salvar_grafico(dados_pn_hard, p_conf["xlabel"], r"$P(N)$", p_conf["label"], 5e-7, 
                           f"{p_nome}_PN_eta_{corte_str}.pdf", y_log=True, leg_title=rf"$|\eta| < {corte_eta}$ (Hard)", subpasta=sub_dir_hard)
            salvar_grafico(dados_evt_hard, p_conf["xlabel"], "Number of events", p_conf["label"], 0.5, 
                           f"{p_nome}_Eventos_eta_{corte_str}.pdf", y_log=True, leg_title=rf"$|\eta| < {corte_eta}$ (Hard)", subpasta=sub_dir_hard)

msg_fim = f"\n{'='*85}\nProcessamento, testes de validação e geração dos plots de Multiplicidade concluídos!\nDetalhes salvos em 'validacao_simulacao_PN.log'\n{'='*85}"
log_print(msg_fim)

log_file.close()