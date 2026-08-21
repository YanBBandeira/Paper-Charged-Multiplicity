import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.ROOT)

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================
casos = {
    "All Physics": ({"cor": "black",       "marker": "o", "pref_soft": "Soft_14TeV_all",      "pref_hard": "Hard_14TeV_all"}),
    "No MPI":      ({"cor": "firebrick",   "marker": "s", "pref_soft": "Soft_14TeV_noMPI",    "pref_hard": "Hard_14TeV_noMPI"}),
    "Only MPI":    ({"cor": "royalblue",   "marker": "^", "pref_soft": "Soft_14TeV_onlyMPI",  "pref_hard": "Hard_14TeV_onlyMPI"}),
    "Only ISR":    ({"cor": "forestgreen", "marker": "v", "pref_soft": "Soft_14TeV_onlyISR",  "pref_hard": "Hard_14TeV_onlyISR"}),
    "Only FSR":    ({"cor": "darkorange",  "marker": "D", "pref_soft": "Soft_14TeV_onlyFSR",  "pref_hard": "Hard_14TeV_onlyFSR"}),
    "No Extra":    ({"cor": "purple",      "marker": "p", "pref_soft": "Soft_14TeV_noExtra",  "pref_hard": "Hard_14TeV_noExtra"})
}

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
bin_width = 0.1

log_file = open("validacao_correlacao.log", "w", encoding="utf-8")

def log_print(texto):
    print(texto)
    log_file.write(texto + "\n")

# =============================================================================
# 2. FUNÇÃO DE PROCESSAMENTO E VALIDAÇÃO
# =============================================================================
def processar_e_validar_dados(arq_mult, arq_eta):
    if not os.path.exists(arq_mult) or not os.path.exists(arq_eta):
        return None, "Arquivos ausentes"

    dados_mult = np.loadtxt(arq_mult)
    N_valores = dados_mult[:, 0]
    counts = dados_mult[:, 1]
    
    total_eventos = np.sum(counts)
    if total_eventos == 0:
        return None, "Zero eventos"
        
    P_N = counts / total_eventos
    media_N = np.sum(N_valores * P_N)
    soma_prob = np.sum(P_N)

    dados_eta = np.loadtxt(arq_eta)
    soma_particulas = dados_eta[:, 1]
    dndeta_dist = soma_particulas / (total_eventos * bin_width)
    integral_dndeta = np.sum(dndeta_dist * bin_width)

    status = "OK"
    if not np.isclose(soma_prob, 1.0, atol=1e-3):
        status = "Erro P(N)!=1"
    elif not np.isclose(media_N, integral_dndeta, atol=0.2):
        status = "Aviso Overflow"

    resultado = {
        "N": N_valores,
        "media_N": media_N,
        "soma_prob": soma_prob,
        "integral": integral_dndeta
    }
    return resultado, status

# =============================================================================
# 3. LOOP PRINCIPAL DE PLOTAGEM
# =============================================================================
for corte_eta in cortes_eta:
    corte_str = f"{corte_eta:.1f}".replace(".", "p")
    delta_eta = 2.0 * corte_eta # Largura total da janela de pseudorapidez
    
    log_print(f"\n{'='*75}\nPROCESSANDO |eta| < {corte_eta}\n{'='*75}")
    
    # Inicializa as três figuras independentes com a mesma estrutura de correlação
    plt.figure(1, figsize=(9, 8)) # Figura 1: dN/deta / <dN/deta>
    max_x1, max_y1 = 0, 0

    plt.figure(2, figsize=(9, 8)) # Figura 2: N * <N> (Produto)
    max_x2, max_y2 = 0, 0

    plt.figure(3, figsize=(9, 8)) # Figura 3: N / <N> (Multiplicidade)
    max_x3, max_y3 = 0, 0

    for nome_caso, config in casos.items():
        pasta_soft = config["pref_soft"].split("_")[-1]
        pasta_hard = config["pref_hard"].split("_")[-1]
        
        arq_mult_ch = os.path.join("..", "Dados", "Soft", pasta_soft, f"{config['pref_soft']}_mult_ch_eta_{corte_str}.dat")
        arq_eta_ch  = os.path.join("..", "Dados", "Soft", pasta_soft, f"{config['pref_soft']}_dndeta_ch_eta_{corte_str}.dat")
        arq_mult_d  = os.path.join("..", "Dados", "Hard", pasta_hard, f"{config['pref_hard']}_mult_d_eta_{corte_str}.dat")
        arq_eta_d   = os.path.join("..", "Dados", "Hard", pasta_hard, f"{config['pref_hard']}_dndeta_d_eta_{corte_str}.dat")
        
        res_ch, status_ch = processar_e_validar_dados(arq_mult_ch, arq_eta_ch)
        res_d, status_d   = processar_e_validar_dados(arq_mult_d, arq_eta_d)

        if res_ch is not None and res_d is not None:
            val_ch, val_D = res_ch['N'], res_d['N']
            media_ch_n, media_D_n = res_ch['media_N'], res_d['media_N']
            min_len_n = min(len(val_ch), len(val_D))

            rotulo = f"{nome_caso}"

            # --- FIGURA 1: dN/deta / <dN/deta> (Estrutura de correlação em densidade) ---
            dndeta_ch_vals = val_ch[:min_len_n] / delta_eta
            dndeta_d_vals  = val_D[:min_len_n] / delta_eta
            mean_dndeta_ch = media_ch_n / delta_eta
            mean_dndeta_d  = media_D_n / delta_eta

            x_norm1 = dndeta_ch_vals / mean_dndeta_ch if mean_dndeta_ch > 0 else dndeta_ch_vals
            y_norm1 = dndeta_d_vals  / mean_dndeta_d  if mean_dndeta_d  > 0 else dndeta_d_vals

            if len(x_norm1) > 0: max_x1 = max(max_x1, x_norm1.max())
            if len(y_norm1) > 0: max_y1 = max(max_y1, y_norm1.max())

            plt.figure(1)
            plt.plot(x_norm1, y_norm1, color=config["cor"], marker=config["marker"], 
                     linestyle='-', linewidth=1.5, markersize=6, alpha=0.85, label=rotulo)

            # --- FIGURA 2: Produto N * <N> ---
            x_prod = val_ch[:min_len_n] * media_ch_n
            y_prod = val_D[:min_len_n] * media_D_n

            if len(x_prod) > 0: max_x2 = max(max_x2, x_prod.max())
            if len(y_prod) > 0: max_y2 = max(max_y2, y_prod.max())

            plt.figure(2)
            plt.plot(x_prod, y_prod, color=config["cor"], marker=config["marker"], 
                     linestyle='-', linewidth=1.5, markersize=6, alpha=0.85, label=rotulo)

            # --- FIGURA 3: Auto-Normalizado N / <N> ---
            x_norm3 = val_ch[:min_len_n] / media_ch_n if media_ch_n > 0 else val_ch[:min_len_n]
            y_norm3 = val_D[:min_len_n]  / media_D_n  if media_D_n  > 0 else val_D[:min_len_n]

            if len(x_norm3) > 0: max_x3 = max(max_x3, x_norm3.max())
            if len(y_norm3) > 0: max_y3 = max(max_y3, y_norm3.max())

            plt.figure(3)
            plt.plot(x_norm3, y_norm3, color=config["cor"], marker=config["marker"], 
                     linestyle='-', linewidth=1.5, markersize=6, alpha=0.85, label=rotulo)
            
            log_print(f"{nome_caso:<15} | <N>_ch: {media_ch_n:.2f} | Status: OK")
        else:
            log_print(f"{nome_caso:<15} | Arquivos ausentes ou inválidos, pulando...")

    dir_fig = os.path.join("Figures", "Correlations")
    os.makedirs(dir_fig, exist_ok=True)

    # =========================================================================
    # SALVAR FIGURA 1: dN/deta / <dN/deta>
    # =========================================================================
    plt.figure(1)
    plt.plot([0, max_x1*1.1], [0, max_x1*1.1], 'k--', alpha=0.3, label="Linear scaling")
    plt.xlim(0, max_x1 * 1.05 if max_x1 > 0 else 1)
    plt.ylim(0, max_y1 * 1.05 if max_y1 > 0 else 1)
    plt.xlabel(r"$(\mathrm{d}N_{ch}/\mathrm{d}\eta) / \langle \mathrm{d}N_{ch}/\mathrm{d}\eta \rangle$")
    plt.ylabel(r"$(\mathrm{d}N_{D}/\mathrm{d}\eta) / \langle \mathrm{d}N_{D}/\mathrm{d}\eta \rangle$")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", frameon=False, title=rf"$|\eta| < {corte_eta}$", title_fontsize=11, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_fig, f"SelfNormalized_dNdeta_eta_{corte_str}.pdf"), bbox_inches="tight")
    plt.close(1)

    # =========================================================================
    # SALVAR FIGURA 2: Produto (N * <N>)
    # =========================================================================
    plt.figure(2)
    plt.xlim(0, max_x2 * 1.05 if max_x2 > 0 else 1)
    plt.ylim(0, max_y2 * 1.05 if max_y2 > 0 else 1)
    plt.xlabel(r"$N_{ch} \cdot \langle N_{ch} \rangle$")
    plt.ylabel(r"$N_{D} \cdot \langle N_{D} \rangle$")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", frameon=False, title=rf"$|\eta| < {corte_eta}$", title_fontsize=11, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_fig, f"Product_Yield_eta_{corte_str}.pdf"), bbox_inches="tight")
    plt.close(2)

    # =========================================================================
    # SALVAR FIGURA 3: Auto-Normalizado (N / <N>)
    # =========================================================================
    plt.figure(3)
    plt.plot([0, max_x3*1.1], [0, max_x3*1.1], 'k--', alpha=0.3, label="Linear scaling")
    plt.xlim(0, max_x3 * 1.05 if max_x3 > 0 else 1)
    plt.ylim(0, max_y3 * 1.05 if max_y3 > 0 else 1)
    plt.xlabel(r"$N_{ch} / \langle N_{ch} \rangle$")
    plt.ylabel(r"$N_{D} / \langle N_{D} \rangle$")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", frameon=False, title=rf"$|\eta| < {corte_eta}$", title_fontsize=11, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_fig, f"SelfNormalized_Yield_eta_{corte_str}.pdf"), bbox_inches="tight")
    plt.close(3)

log_print(f"\n{'='*75}\nProcesso de geração das três figuras finalizado com sucesso!\n{'='*75}")
log_file.close()