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

# Mapeamento das configurações de simulação (Soft vs Hard)
casos = {
    "All Physics": ("Soft_14TeV_all", "Hard_14TeV_all"),
    "No MPI":      ("Soft_14TeV_noMPI", "Hard_14TeV_noMPI"),
    "Only MPI":    ("Soft_14TeV_onlyMPI", "Hard_14TeV_onlyMPI"),
    "Only ISR":    ("Soft_14TeV_onlyISR", "Hard_14TeV_onlyISR"),
    "Only FSR":    ("Soft_14TeV_onlyFSR", "Hard_14TeV_onlyFSR"),
    "No Extra":    ("Soft_14TeV_noExtra", "Hard_14TeV_noExtra"),
}

# Partículas extraídas nas simulações
particulas_info = {
    "Charged": {
        "sigla": "ch", 
        "label": r"Charged Particles ($h^{\pm}$)",
        "ylabel": r"$dN/d\eta$",
    },
    "D0_mesons": {
        "sigla": "d0",
        "label": r"$D^0$ Mesons",
        "ylabel": r"$dN/d\eta$",
    },
    "D_mesons": {
        "sigla": "d",
        "label": r"$D^\pm$ Mesons",
        "ylabel": r"$dN/d\eta$",
    },
    "Ds_mesons": {
        "sigla": "ds",
        "label": r"$D_s$ Mesons",
        "ylabel": r"$dN/d\eta$",
    },
}

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

# Faixas de pT exatamente idênticas às usadas na geração dos dados
ranges_pt = [
    (0.0, None,  "global",   "global"),          # Caso global (pT > 0)
    (1.0, 2.0,   "pt_1_2",   "1.0 < p_T < 2.0"),  # 1 < pT < 2 GeV/c
    (2.0, 4.0,   "pt_2_4",   "2.0 < p_T < 4.0"),  # 2 < pT < 4 GeV/c
    (4.0, 6.0,   "pt_4_6",   "4.0 < p_T < 6.0"),  # 4 < pT < 6 GeV/c
    (6.0, 8.0,   "pt_6_8",   "6.0 < p_T < 8.0"),  # 6 < pT < 8 GeV/c
    (8.0, 12.0,  "pt_8_12",  "8.0 < p_T < 12.0"), # 8 < pT < 12 GeV/c
    (12.0, 24.0, "pt_12_24", "12.0 < p_T < 24.0"),# 12 < pT < 24 GeV/c
]

log_file = open("validacao_simulacao_dndeta.log", "w", encoding="utf-8")

def log_print(texto, file=log_file):
    """Escreve no console e no arquivo de log simultaneamente."""
    print(texto)
    file.write(texto + "\n")

# =============================================================================
# 2. FUNÇÃO DE LEITURA, NORMALIZAÇÃO E CHECAGEM DE SANIDADE
# =============================================================================
def processar_e_normalizar_dndeta(arq_path, arq_mult, nome_caso, nome_particula, tag_corte):
    """
    Lê o arquivo .dat de dN/deta, extrai o número de eventos do histograma de multiplicidade,
    normaliza os valores por evento e largura de bin, faz a checagem de sanidade 
    usando a soma correta dos bins e só libera o resultado se tudo estiver correto.
    """
    if not os.path.exists(arq_path):
        return None, "Arquivo ausente"

    dados = np.loadtxt(arq_path)
    if dados.ndim == 1:
        dados = dados.reshape(1, -1)
    
    if len(dados) == 0:
        return None, "Arquivo vazio"
        
    eta_vals = dados[:, 0]  # Centro do bin de eta
    dndeta_bruto = dados[:, 1] # Valor bruto acumulado
    
    # Determina a largura do bin para normalização correta do dN/deta
    if len(eta_vals) > 1:
        largura_bin = eta_vals[1] - eta_vals[0]
    else:
        largura_bin = 0.1

    # Obtém o número de eventos (N_ev) somando os pesos do histograma de multiplicidade
    n_eventos = 0.0
    media_mult_direta = None
    if os.path.exists(arq_mult):
        dados_m = np.loadtxt(arq_mult)
        if dados_m.ndim == 1 and len(dados_m) >= 2:
            dados_m = dados_m.reshape(1, -1)
        
        x_m = dados_m[:, 0]
        y_m = dados_m[:, 1]
        n_eventos = np.sum(y_m)
        if n_eventos > 0:
            media_mult_direta = np.sum(x_m * y_m) / n_eventos

    if n_eventos <= 0:
        log_print(f"  [AVISO | {nome_caso} | {nome_particula} | {tag_corte}]: Número de eventos inválido ou zero encontrado!")
        return None, "Eventos inválidos"

    # NORMALIZAÇÃO: Divide pelo número de eventos e pela largura do bin
    dndeta_normalizado = dndeta_bruto / (n_eventos * largura_bin)

    # 3. CHECAGEM DE SANIDADE (Soma discreta exata dos bins do histograma)
    integral_dndeta = np.sum(dndeta_normalizado) * largura_bin
    
    msg = f"  [Sanidade OK | {nome_caso} | {nome_particula} | {tag_corte}]: N_ev = {int(n_eventos)} | Integral dN/deta = {integral_dndeta:.4f}"
    if media_mult_direta is not None:
        diff_perc = abs(integral_dndeta - media_mult_direta) / media_mult_direta * 100 if media_mult_direta > 0 else 0
        msg += f" | Média <N> (Mult) = {media_mult_direta:.4f} (Dif: {diff_perc:.2f}%)"
    
    log_print(msg)

    resultado = {
        "eta": eta_vals,
        "dndeta": dndeta_normalizado,
    }
    return resultado, "OK"

# =============================================================================
# 3. FUNÇÃO DE SALVAR GRÁFICO DE dN/deta
# =============================================================================
def salvar_grafico_dndeta(dados_plot, xlabel, ylabel, label, nome_pdf, leg_title="", subpasta=""):
    """
    Gera e salva os gráficos comparativos de dN/deta usando o mplhep.
    """
    caminho_dir = os.path.join("Figures", subpasta)
    os.makedirs(caminho_dir, exist_ok=True)
    caminho_completo = os.path.join(caminho_dir, nome_pdf)

    plt.figure(figsize=(9, 6))
    max_y_global = 0

    for item in dados_plot:
        x_vals = item["x"]
        y_vals = np.array(item["y"], dtype=float)
        
        if len(x_vals) == 0: continue

        if len(x_vals) > 1:
            largura_bin = x_vals[1] - x_vals[0]
        else:
            largura_bin = 0.1
            
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

    plt.yscale("linear")
    top_lim = max_y_global * 1.4 if max_y_global > 0 else 1.0
    plt.ylim(bottom=0, top=top_lim)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(0.05, 0.88, label, transform=plt.gca().transAxes, fontsize=12, verticalalignment="center")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize=9, title=leg_title, title_fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig(caminho_completo, bbox_inches="tight")
    plt.close()

# =============================================================================
# 4. LOOP PRINCIPAL (ETA, PT, PARTÍCULAS E CASOS)
# =============================================================================
for corte_eta in cortes_eta:
    corte_eta_str = f"{corte_eta:.1f}".replace(".", "p")
    
    for pt_min, pt_max, pt_tag, pt_leg_desc in ranges_pt:
        if pt_max is None:
            tipo_corte_dir = f"eta_{corte_eta_str}/pt_global"
            # O gerador salva o caso global de dN/deta sem o sufixo _global,
            # mas a multiplicidade global continua com _global.
            tag_arquivo_dndeta = f"eta_{corte_eta_str}"
            tag_arquivo_mult = f"eta_{corte_eta_str}_global"
            legenda_corte = rf"$|\eta| < {corte_eta}$ $p_T > 0$ GeV/$c$ "
        else:
            tipo_corte_dir = f"eta_{corte_eta_str}/{pt_tag}"
            tag_arquivo_dndeta = f"eta_{corte_eta_str}_{pt_tag}"
            tag_arquivo_mult = f"eta_{corte_eta_str}_{pt_tag}"
            legenda_corte = rf"$|\eta| < {corte_eta},\ {pt_leg_desc}$ GeV/$c$"

        msg_separador = f"\n{'='*85}\n PROCESSANDO, NORMALIZANDO E VALIDANDO: |eta| < {corte_eta} | pT: {pt_tag}\n{'='*85}"
        log_print(msg_separador)

        for p_nome, p_conf in particulas_info.items():
            sigla = p_conf["sigla"]
            
            dados_dndeta_soft = []
            dados_dndeta_hard = []

            for idx, (nome_caso, (pref_soft, pref_hard)) in enumerate(casos.items()):
                pasta_soft = pref_soft.split("_")[-1]
                pasta_hard = pref_hard.split("_")[-1]
                
                cor = cores[idx % len(cores)]
                estilo = estilos_linha[idx % len(estilos_linha)]

                # -------------------------------------------------------------
                # 4.1. Processamento, Normalização e Sanidade Soft
                # -------------------------------------------------------------
                arq_soft = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_dndeta_{sigla}_{tag_arquivo_dndeta}.dat")
                arq_mult_soft = os.path.join("..", "Dados", "Soft", pasta_soft, f"{pref_soft}_mult_{sigla}_{tag_arquivo_mult}.dat")

                res_s, status_s = processar_e_normalizar_dndeta(
                    arq_soft, arq_mult_soft, f"Soft - {nome_caso}", p_nome, tag_arquivo_dndeta
                )

                if res_s is not None:
                    dados_dndeta_soft.append({
                        "x": res_s["eta"], 
                        "y": res_s["dndeta"], 
                        "label": f"{nome_caso}", 
                        "color": cor, 
                        "ls": estilo
                    })

                # -------------------------------------------------------------
                # 4.2. Processamento, Normalização e Sanidade Hard
                # -------------------------------------------------------------
                arq_hard = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_dndeta_{sigla}_{tag_arquivo_dndeta}.dat")
                arq_mult_hard = os.path.join("..", "Dados", "Hard", pasta_hard, f"{pref_hard}_mult_{sigla}_{tag_arquivo_mult}.dat")

                res_h, status_h = processar_e_normalizar_dndeta(
                    arq_hard, arq_mult_hard, f"Hard - {nome_caso}", p_nome, tag_arquivo_dndeta
                )

                if res_h is not None:
                    dados_dndeta_hard.append({
                        "x": res_h["eta"], 
                        "y": res_h["dndeta"], 
                        "label": f"{nome_caso}", 
                        "color": cor, 
                        "ls": estilo
                    })

            # =================================================================
            # 4.3. GERAÇÃO DOS PLOTS DE dN/deta NORMALIZADOS
            # =================================================================
            if dados_dndeta_soft:
                sub_dir_soft = os.path.join("Soft", tipo_corte_dir, p_nome)
                salvar_grafico_dndeta(
                    dados_dndeta_soft, 
                    r"Pseudorapidity ($\eta$)", 
                    p_conf["ylabel"], 
                    p_conf["label"], 
                    f"{p_nome}_dNdEta.pdf", 
                    leg_title=legenda_corte, 
                    subpasta=sub_dir_soft
                )

            if dados_dndeta_hard:
                sub_dir_hard = os.path.join("Hard", tipo_corte_dir, p_nome)
                salvar_grafico_dndeta(
                    dados_dndeta_hard, 
                    r"Pseudorapidity ($\eta$)", 
                    p_conf["ylabel"], 
                    p_conf["label"], 
                    f"{p_nome}_dNdEta.pdf", 
                    leg_title=legenda_corte, 
                    subpasta=sub_dir_hard
                )

msg_fim = f"\n{'='*85}\nProcesso concluído com sucesso! Verifique os gráficos e o log de auditoria.\n{'='*85}"
log_print(msg_fim)
log_file.close()