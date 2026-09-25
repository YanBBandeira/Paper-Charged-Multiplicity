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

# Apenas as partículas que são efetivamente extraídas
particulas_info = {
    "Charged": {
        "sigla": "ch", 
        "label": r"Charged Particles ($h^{\pm}$)",
        "xlabel": r"$N_{\mathrm{ch}}$",
    },
    "D0_mesons": {
        "sigla": "d0",
        "label": r"$D^0$ Mesons",
        "xlabel": r"$N_{D^0}$",
    },
    "D_mesons": {
        "sigla": "d",
        "label": r"$D^\pm$ Mesons",
        "xlabel": r"$N_{D}$",
    },
    "Ds_mesons": {
        "sigla": "ds",
        "label": r"$D_s$ Mesons",
        "xlabel": r"$N_{D_s}$",
    },
}

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

# Definição dos ranges de pT (pt_min, pt_max). 
# Se pt_max for negativo, assume-se o caso global (sem corte superior de pT).
# Definição correta dos ranges de pT exatamente como no script de simulação C++
ranges_pt = [
    (0.0, -1.0, "global",      "global"),           # Caso global (pT > 0)
    (1.0,  2.0, "pt_1_2",      "pt_1_2"),           # 1 < pT < 2 GeV/c[cite: 1]
    (2.0,  4.0, "pt_2_4",      "pt_2_4"),           # 2 < pT < 4 GeV/c[cite: 1]
    (4.0,  6.0, "pt_4_6",      "pt_4_6"),           # 4 < pT < 6 GeV/c[cite: 1]
    (6.0,  8.0, "pt_6_8",      "pt_6_8"),           # 6 < pT < 8 GeV/c[cite: 1]
    (8.0, 12.0, "pt_8_12",     "pt_8_12"),          # 8 < pT < 12 GeV/c[cite: 1]
    (12.0, 24.0, "pt_12_24",   "pt_12_24"),         # 12 < pT < 24 GeV/c[cite: 1]
]

# Abre o arquivo de log no modo de escrita na raiz
log_file = open("validacao_simulacao_PN.log", "w", encoding="utf-8")

def log_print(texto, file=log_file):
    """Escreve no console e no arquivo de log simultaneamente."""
    print(texto)
    file.write(texto + "\n")

# =============================================================================
# 2. FUNÇÃO DE LEITURA E VALIDAÇÃO DA MULTIPLICIDADE
# =============================================================================
def processar_e_validar_dados(arq_mult):
    """
    Lê o arquivo de multiplicidade .dat, remove apenas o underflow (1ª linha), 
    executa os testes de sanidade e retorna os dados processados.
    """
    if not os.path.exists(arq_mult):
        return None, "Arquivo ausente"

    # Leitura da Multiplicidade
    dados_mult = np.loadtxt(arq_mult)
    if dados_mult.ndim == 1:
        dados_mult = dados_mult.reshape(1, -1)
    
    # Remove apenas o underflow (1ª linha), mantendo o overflow e o restante
    if len(dados_mult) > 1:
        dados_mult = dados_mult[1:]
    
    N_valores = dados_mult[:, 0]  # Eixo X
    counts = dados_mult[:, 1]     # Eixo Y (Eventos)
    
    total_eventos = np.sum(counts)
    if total_eventos == 0:
        return None, "Zero eventos"
        
    P_N = counts / total_eventos
    media_N = np.sum(N_valores * P_N)
    soma_prob = np.sum(P_N)

    # Status dos Testes
    status = "OK"
    if not np.isclose(soma_prob, 1.0, atol=1e-3):
        status = "Erro P(N)!=1"

    resultado = {
        "N": N_valores,
        "counts": counts,
        "P_N": P_N,
        "media_N": media_N,
        "soma_prob": soma_prob,
    }
    return resultado, status

# =============================================================================
# 3. FUNÇÃO DE SALVAR GRÁFICO (COM HEADROOM NO EIXO Y)
# =============================================================================
def salvar_grafico(dados_plot, xlabel, ylabel, label, y_min, nome_pdf, y_log=True, leg_title="", subpasta=""):
    """
    Salva os gráficos comparativos utilizando o hep.histplot() do mplhep.
    Estrutura de saída: Figures/{subpasta}/{nome_pdf}
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
        plt.xlim(left=0, right=max_x_global * 1.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(0.4, 0.9, label, transform=plt.gca().transAxes, fontsize=12, verticalalignment="center")
    plt.grid(True, which="both", linestyle="--", alpha=0.2)
    plt.legend(loc="best", fontsize=10, title=leg_title, title_fontsize=11, frameon=False)

    plt.tight_layout()
    plt.savefig(caminho_completo, bbox_inches="tight")
    plt.close()

# =============================================================================
# 4. LOOP PRINCIPAL (ETA, PT, PARTÍCULAS E CASOS)
# =============================================================================
for corte_eta in cortes_eta:
    corte_eta_str = f"{corte_eta:.1f}".replace(".", "p")
    
    for pt_min, pt_max, pt_tag, _ in ranges_pt:
        # Ajustado para seguir exatamente a convenção da geração em Pythia:
        # ex.: ..._mult_ch_eta_0p5_pt_1_2.dat
        if pt_max < 0:
            tipo_corte_dir = f"eta_{corte_eta_str}/pt_global"
            tag_arquivo = f"eta_{corte_eta_str}_global"
            legenda_corte = rf"$|\eta| < {corte_eta}$ $p_T > 0$ GeV/$c$"
        else:
            tipo_corte_dir = f"eta_{corte_eta_str}/{pt_tag}"
            tag_arquivo = f"eta_{corte_eta_str}_{pt_tag}"
            legenda_corte = rf"$|\eta| < {corte_eta},\ {pt_min} < p_T < {pt_max}$ GeV/c"

        msg_separador = f"\n{'='*85}\n PROCESSANDO: |eta| < {corte_eta} | pT: {pt_tag}\n{'='*85}"
        log_print(msg_separador)

        for p_nome, p_conf in particulas_info.items():
            sigla = p_conf["sigla"]
            
            dados_pn_soft, dados_evt_soft = [], []
            dados_pn_hard, dados_evt_hard = [], []

            cabecalho = f"\n--- Partícula: {p_nome} ---\n" + \
                        f"{'Configuração':<15} | {'Processo':<6} | {'Soma P(N)':<10} | {'<N>':<8} | {'Status'}\n" + \
                        "-" * 55
            log_print(cabecalho)

            for idx, (nome_caso, (pref_soft, pref_hard)) in enumerate(casos.items()):
                pasta_soft = pref_soft.split("_")[-1]
                pasta_hard = pref_hard.split("_")[-1]
                
                cor = cores[idx % len(cores)]
                estilo = estilos_linha[idx % len(estilos_linha)]

                # -------------------------------------------------------------
                # 4.1. Leitura e Validação: Soft
                # -------------------------------------------------------------
                arq_mult_soft = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_mult_{sigla}_{tag_arquivo}.dat")
                res_s, status_s = processar_e_validar_dados(arq_mult_soft)

                if res_s is not None:
                    linha_s = f"{nome_caso:<15} | {'Soft':<6} | {res_s['soma_prob']:<10.4f} | {res_s['media_N']:<8.4f} | {status_s}"
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

                # -------------------------------------------------------------
                # 4.2. Leitura e Validação: Hard
                # -------------------------------------------------------------
                arq_mult_hard = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_mult_{sigla}_{tag_arquivo}.dat")
                res_h, status_h = processar_e_validar_dados(arq_mult_hard)

                if res_h is not None:
                    linha_h = f"{nome_caso:<15} | {'Hard':<6} | {res_h['soma_prob']:<10.4f} | {res_h['media_N']:<8.4f} | {status_h}"
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

            # =================================================================
            # 4.3. GERAÇÃO DOS PLOTS SALVOS NA HIERARQUIA DE PASTAS
            # =================================================================
            if dados_pn_soft:
                sub_dir_soft = os.path.join("Soft", tipo_corte_dir, p_nome)
                salvar_grafico(dados_pn_soft, p_conf["xlabel"], r"$P(N)$", p_conf["label"], 5e-7, 
                               f"{p_nome}_PN.pdf", y_log=True, leg_title=legenda_corte, subpasta=sub_dir_soft)
                salvar_grafico(dados_evt_soft, p_conf["xlabel"], "Number of events", p_conf["label"], 0.5, 
                               f"{p_nome}_Eventos.pdf", y_log=True, leg_title=legenda_corte, subpasta=sub_dir_soft)

            if dados_pn_hard:
                sub_dir_hard = os.path.join("Hard", tipo_corte_dir, p_nome)
                salvar_grafico(dados_pn_hard, p_conf["xlabel"], r"$P(N)$", p_conf["label"], 5e-7, 
                               f"{p_nome}_PN.pdf", y_log=True, leg_title=legenda_corte, subpasta=sub_dir_hard)
                salvar_grafico(dados_evt_hard, p_conf["xlabel"], "Number of events", p_conf["label"], 0.5, 
                               f"{p_nome}_Eventos.pdf", y_log=True, leg_title=legenda_corte, subpasta=sub_dir_hard)

msg_fim = f"\n{'='*85}\nProcessamento e geração de todos os plots de Multiplicidade concluídos!\nLogs salvos em 'validacao_simulacao_PN.log'\n{'='*85}"
log_print(msg_fim)

log_file.close()