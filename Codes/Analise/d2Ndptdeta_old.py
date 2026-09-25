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
    },
    "Kaons": {
        "sigla": "k",
        "label": r"Kaons",
    },
    "D_mesons": {
        "sigla": "d",
        "label": r"$D$ Mesons",
    },
    "Ds_mesons": {
        "sigla": "ds",
        "label": r"$D_s$ Mesons",
    },
}

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

# Abre o arquivo de log no modo de escrita na raiz para auditoria da normalização 2D
log_file = open("validacao_simulacao_d2n.log", "w", encoding="utf-8")

def log_print(texto, file=log_file):
    """Escreve no console e no arquivo de log simultaneamente."""
    print(texto)
    file.write(texto + "\n")

# =============================================================================
# 2. FUNÇÃO DE LEITURA, VALIDAÇÃO CRUZADA E INTEGRAÇÃO 2D
# =============================================================================
def processar_e_validar_matriz_2d(arq_d2n, arq_mult):
    """
    Lê a matriz 2D tabulada [eta, pt, d2N], calcula sua integral volumétrica 
    e a compara com a média obtida no arquivo de multiplicidade para testar a normalização.
    """
    if not os.path.exists(arq_d2n) or not os.path.exists(arq_mult):
        return None, "Arquivos ausentes"

    # Leitura dos dados 2D
    dados_raw = np.loadtxt(arq_d2n)
    if dados_raw.size == 0:
        return None, "Matriz vazia"

    etad = np.unique(dados_raw[:, 0])
    ptd = np.unique(dados_raw[:, 1])
    
    n_eta = len(etad)
    n_pt = len(ptd)
    
    if n_eta < 2 or n_pt < 2:
        return None, "Bins insuficientes"

    d_eta = etad[1] - etad[0]
    d_pt = ptd[1] - ptd[0]
    
    matriz_2d = dados_raw[:, 2].reshape(n_eta, n_pt)
    
    # Integral 2D: Soma( d2N/(deta dpt) * d_eta * d_pt )
    integral_2d = np.sum(matriz_2d) * d_eta * d_pt

    # Leitura do arquivo de multiplicidade correspondente para referência cruzada da média
    dados_mult = np.loadtxt(arq_mult)
    if dados_mult.ndim == 1:
        dados_mult = dados_mult.reshape(1, -1)
    
    N_valores = dados_mult[:, 0]
    counts = dados_mult[:, 1]
    total_eventos = np.sum(counts)
    
    if total_eventos == 0:
        return None, "Zero eventos"
        
    media_N_teorica = np.sum(N_valores * (counts / total_eventos))

    # Teste de Sanidade / Normalização
    status = "OK"
    if not np.isclose(integral_2d, media_N_teorica, rtol=1e-2, atol=1e-2):
        status = "Erro Normalização/Integral"

    resultado = {
        "eta": etad,
        "pt": ptd,
        "matriz": matriz_2d,
        "integral_2d": integral_2d,
        "media_N": media_N_teorica
    }
    return resultado, status

# =============================================================================
# 3. FUNÇÃO DE SALVAR PROJEÇÃO
# =============================================================================
def salvar_projecao_d2n(dados_plot, xlabel, ylabel, label, nome_pdf, y_log=True, leg_title="", subpasta=""):
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
        plt.ylim(bottom=1e-6, top=max_y_global * 100 if max_y_global > 0 else 1.0)
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
# 4. LOOP PRINCIPAL DE ANÁLISE, VALIDAÇÃO E PLOTAGEM DAS MATRIZES 2D
# =============================================================================
for corte_eta in cortes_eta:
    corte_str = f"{corte_eta:.1f}".replace(".", "p")
    
    msg_separador = f"\n{'='*85}\n PROCESSANDO E VALIDANDO MATRIZES 2D EM |eta| < {corte_eta}\n{'='*85}"
    log_print(msg_separador)

    for p_nome, p_conf in particulas_info.items():
        sigla = p_conf["sigla"]
        
        dados_d2n_soft = []
        dados_d2n_hard = []

        cabecalho = f"\n--- Partícula: {p_nome} ---\n" + \
                    f"{'Configuração':<15} | {'Processo':<6} | {'Integral 2D':<12} | {'<N> Esperado':<12} | {'Status'}\n" + \
                    "-" * 60
        log_print(cabecalho)

        for idx, (nome_caso, (pref_soft, pref_hard)) in enumerate(casos.items()):
            pasta_soft = pref_soft.split("_")[-1]
            pasta_hard = pref_hard.split("_")[-1]
            
            cor = cores[idx % len(cores)]
            estilo = estilos_linha[idx % len(estilos_linha)]

            # -----------------------------------------------------------------
            # 4.1. Validação Soft
            # -----------------------------------------------------------------
            arq_d2n_soft  = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_d2n_etapt_{sigla}_eta_{corte_str}.dat")
            arq_mult_soft = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_mult_{sigla}_eta_{corte_str}.dat")
            res_s, status_s = processar_e_validar_matriz_2d(arq_d2n_soft, arq_mult_soft)

            if res_s is not None:
                linha_s = f"{nome_caso:<15} | {'Soft':<6} | {res_s['integral_2d']:<12.4f} | {res_s['media_N']:<12.4f} | {status_s}"
                log_print(linha_s)
                
                dados_d2n_soft.append({
                    "eta": res_s["eta"], "pt": res_s["pt"], "matriz": res_s["matriz"],
                    "label": nome_caso, "color": cor, "ls": estilo
                })

            # -----------------------------------------------------------------
            # 4.2. Validação Hard
            # -----------------------------------------------------------------
            arq_d2n_hard  = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_d2n_etapt_{sigla}_eta_{corte_str}.dat")
            arq_mult_hard = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_mult_{sigla}_eta_{corte_str}.dat")
            res_h, status_h = processar_e_validar_matriz_2d(arq_d2n_hard, arq_mult_hard)

            if res_h is not None:
                linha_h = f"{nome_caso:<15} | {'Hard':<6} | {res_h['integral_2d']:<12.4f} | {res_h['media_N']:<12.4f} | {status_h}"
                log_print(linha_h)

                dados_d2n_hard.append({
                    "eta": res_h["eta"], "pt": res_h["pt"], "matriz": res_h["matriz"],
                    "label": nome_caso, "color": cor, "ls": estilo
                })

        sub_dir_soft = os.path.join("Soft_QCD", p_nome, "2D_Analysis")
        sub_dir_hard = os.path.join("Hard_QCD", p_nome, "2D_Analysis")

        # Projeção em Eta a partir da matriz 2D validada
        proj_eta_soft = []
        for item in dados_d2n_soft:
            proj_y = np.sum(item["matriz"], axis=1) * (item["pt"][1] - item["pt"][0])
            proj_eta_soft.append({"x": item["eta"], "y": proj_y, "label": item["label"], "color": item["color"], "ls": item["ls"]})

        proj_eta_hard = []
        for item in dados_d2n_hard:
            proj_y = np.sum(item["matriz"], axis=1) * (item["pt"][1] - item["pt"][0])
            proj_eta_hard.append({"x": item["eta"], "y": proj_y, "label": item["label"], "color": item["color"], "ls": item["ls"]})

        if proj_eta_soft:
            salvar_projecao_d2n(proj_eta_soft, r"$\eta$", r"$\int \frac{\mathrm{d}^2N}{\mathrm{d}\eta \, \mathrm{d}p_T} \mathrm{d}p_T$", 
                                p_conf["label"], f"{p_nome}_proj_eta_from_matrix_{corte_str}.pdf", y_log=False, 
                                leg_title=rf"$|\eta| < {corte_eta}$ (Soft)", subpasta=sub_dir_soft)

        if proj_eta_hard:
            salvar_projecao_d2n(proj_eta_hard, r"$\eta$", r"$\int \frac{\mathrm{d}^2N}{\mathrm{d}\eta \, \mathrm{d}p_T} \mathrm{d}p_T$", 
                                p_conf["label"], f"{p_nome}_proj_eta_from_matrix_{corte_str}.pdf", y_log=False, 
                                leg_title=rf"$|\eta| < {corte_eta}$ (Hard)", subpasta=sub_dir_hard)

msg_fim = f"\n{'='*85}\nValidação e plotagem das matrizes 2D concluídas com sucesso!\nLog salvo em 'validacao_simulacao_d2n.log'\n{'='*85}"
log_print(msg_fim)
log_file.close()