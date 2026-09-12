import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.ROOT)

# =============================================================================
# 1. CLASSES DE MULTIPLICIDADE OFICIAIS ALICE (CLASSES I A X)
# =============================================================================
classes_alice = [
    ("I",    0.0,   0.95),
    ("II",   0.95,  4.7),
    ("III",  4.7,   9.5),
    ("IV",   9.5,   14.0),
    ("V",    14.0,  19.0),
    ("VI",   19.0,  28.0),
    ("VII",  28.0,  38.0),
    ("VIII", 38.0,  48.0),
    ("IX",   48.0,  68.0),
    ("X",    68.0,  100.0)
]

casos = {
    "All Physics": {
        "pasta": "all",
        "pref_soft": "Soft_14TeV_all",
        "pref_hard": "Hard_14TeV_all"
    }
}

# Configuração de cores e marcadores para os cortes em eta
estilos_eta = {
    0.5: {"cor": "black",       "marker": "o"},
    1.0: {"cor": "firebrick",   "marker": "s"},
    2.0: {"cor": "royalblue",   "marker": "^"},
    3.0: {"cor": "forestgreen", "marker": "v"},
    4.0: {"cor": "darkorange",  "marker": "D"},
    5.0: {"cor": "purple",      "marker": "p"}
}

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

diretorio_script = os.path.dirname(os.path.abspath(__file__))
base_dados_dir   = os.path.abspath(os.path.join(diretorio_script, "..", "Dados"))
base_figuras_dir = os.path.join(diretorio_script, "Figures", "Correlacoes_dNdeta_All")

os.makedirs(base_figuras_dir, exist_ok=True)
log_file_path = os.path.join(diretorio_script, "validacao_correlacoes_dndeta.log")
log_file = open(log_file_path, "w", encoding="utf-8")

def log_print(texto):
    print(texto)
    log_file.write(texto + "\n")

# =============================================================================
# 2. FUNÇÕES DE CÁLCULO DE MÉDIAS DE dN/deta POR PERCENTIL
# =============================================================================
def obter_norm_soft_dndeta(dndeta_ch_soft, classes_info):
    """
    Calcula a média de dN_ch/deta por classe de percentil ALICE.
    """
    n_evts = len(dndeta_ch_soft)
    if n_evts == 0:
        return None, 0.0, []
        
    media_glob = np.mean(dndeta_ch_soft)
    # Ordena do evento de maior dN/deta para o menor
    sort_idx = np.argsort(-dndeta_ch_soft)
    dndeta_sorted = dndeta_ch_soft[sort_idx]
    
    x_norm_list, log_soft = [], []
    for nome_cls, p_inf_pct, p_sup_pct in classes_info:
        i_start = int(np.round((p_inf_pct / 100.0) * n_evts))
        i_end   = max(i_start + 1, int(np.round((p_sup_pct / 100.0) * n_evts)))
        
        faixa = dndeta_sorted[i_start:i_end]
        m_local = np.mean(faixa) if len(faixa) > 0 else 0.0
        norm = m_local / media_glob if media_glob > 0 else 0.0
        
        x_norm_list.append(norm)
        log_soft.append((nome_cls, np.min(faixa), np.max(faixa), m_local, norm))
        
    return np.array(x_norm_list), media_glob, log_soft


def obter_norm_hard_dndeta(dndeta_ch_hard, dndeta_d_hard, classes_info):
    """
    Classifica os eventos Hard pela multiplicidade Soft/Charged dN_ch/deta,
    e calcula a média de dN_D/deta dos mésons D no mesmo percentil.
    """
    n_evts = len(dndeta_ch_hard)
    if n_evts == 0:
        return None, 0.0, []
        
    media_glob_d = np.mean(dndeta_d_hard)
    sort_idx = np.argsort(-dndeta_ch_hard)
    ch_sorted, d_sorted = dndeta_ch_hard[sort_idx], dndeta_d_hard[sort_idx]
    
    y_norm_list, log_hard = [], []
    for nome_cls, p_inf_pct, p_sup_pct in classes_info:
        i_start = int(np.round((p_inf_pct / 100.0) * n_evts))
        i_end   = max(i_start + 1, int(np.round((p_sup_pct / 100.0) * n_evts)))
        
        faixa_ch, faixa_d = ch_sorted[i_start:i_end], d_sorted[i_start:i_end]
        m_local_d = np.mean(faixa_d) if len(faixa_d) > 0 else 0.0
        norm_d = m_local_d / media_glob_d if media_glob_d > 0 else 0.0
        
        y_norm_list.append(norm_d)
        log_hard.append((nome_cls, np.min(faixa_ch), np.max(faixa_ch), m_local_d, norm_d))
        
    return np.array(y_norm_list), media_glob_d, log_hard

# =============================================================================
# 3. PROCESSAMENTO - PLOTS INDIVIDUAIS E ARMAZENAMENTO DAS CURVAS
# =============================================================================
dados_curvas_eta = {}  # Dicionário para guardar (x_norm, y_norm) de cada eta

for corte_eta in cortes_eta:
    corte_str = f"{corte_eta:.1f}".replace(".", "p")
    delta_eta = 2.0 * corte_eta  # Largura total da janela |eta| < corte_eta
    
    log_print(f"\n{'='*120}\nPROCESSANDO CORRELAÇÃO dN/deta | |eta| < {corte_eta} (Delta_eta = {delta_eta:.1f})\n{'='*120}")
    
    config = casos["All Physics"]
    arq_soft = os.path.join(base_dados_dir, "Soft", config["pasta"], f"{config['pref_soft']}_events_eta_{corte_str}.dat")
    arq_hard = os.path.join(base_dados_dir, "Hard", config["pasta"], f"{config['pref_hard']}_events_eta_{corte_str}.dat")
    
    if not os.path.exists(arq_soft) or not os.path.exists(arq_hard):
        log_print(f"[AVISO] Arquivos inexistentes para All Physics em |eta| < {corte_eta}")
        continue
        
    dados_soft = np.loadtxt(arq_soft, skiprows=1)
    dados_hard = np.loadtxt(arq_hard, skiprows=1)
    
    # Extração de N e conversão direta para dN/deta por evento
    n_ch_soft = dados_soft[:, 0] if dados_soft.ndim > 1 else dados_soft
    n_ch_hard, n_d_hard = dados_hard[:, 0], dados_hard[:, 1]

    dndeta_ch_soft = n_ch_soft / delta_eta
    dndeta_ch_hard = n_ch_hard / delta_eta
    dndeta_d_hard  = n_d_hard  / delta_eta

    # Cálculo das médias e normalizações
    x_norm, media_ch_glob, log_soft = obter_norm_soft_dndeta(dndeta_ch_soft, classes_alice)
    y_norm, media_d_glob,  log_hard = obter_norm_hard_dndeta(dndeta_ch_hard, dndeta_d_hard, classes_alice)

    if x_norm is not None and y_norm is not None:
        dados_curvas_eta[corte_eta] = (x_norm, y_norm)

        # Log de Validação com dN/deta
        log_print(f"\n>>> CASO: All Physics | Evts Soft: {len(dndeta_ch_soft)} | Evts Hard: {len(dndeta_ch_hard)}")
        log_print(f"    <<dN_ch/deta>>_soft = {media_ch_glob:.4f} | <<dN_D/deta>>_hard = {media_d_glob:.6f}")
        log_print(f"   {'Classe':<6} | {'dN_ch/deta (Faixa)':<20} | {'<dN_ch/deta>':<14} | {'X_norm (Soft)':<15} | {'dN_ch/deta Hard (Faixa)':<22} | {'<dN_D/deta>':<14} | {'Y_norm (Hard)':<15}")
        log_print("   " + "-"*125)
        
        for i in range(len(classes_alice)):
            c_nome  = classes_alice[i][0]
            faixa_s = f"[{log_soft[i][1]:.2f}, {log_soft[i][2]:.2f}]"
            faixa_h = f"[{log_hard[i][1]:.2f}, {log_hard[i][2]:.2f}]"
            log_print(f"   {c_nome:<6} | {faixa_s:<20} | {log_soft[i][3]:<14.4f} | {log_soft[i][4]:<15.4f} | {faixa_h:<22} | {log_hard[i][3]:<14.6f} | {log_hard[i][4]:<15.4f}")

        # Plot Individual para o corte de eta
        plt.figure(figsize=(8, 7))
        estilo = estilos_eta[corte_eta]
        plt.plot(x_norm, y_norm, color=estilo["cor"], marker=estilo["marker"], 
                 linestyle='-', linewidth=1.5, markersize=7, alpha=0.9, label=f"All Physics ($|\\eta| < {corte_eta}$)")
        
        lim_x = np.nanmax(x_norm) * 1.1
        lim_y = np.nanmax(y_norm) * 1.1
        lim_ref = max(lim_x, lim_y)

        plt.plot([0, lim_ref], [0, lim_ref], 'k--', alpha=0.3, label="Linear scaling")
        plt.xlim(0, lim_x)
        plt.ylim(0, lim_y)
        
        plt.xlabel(r"$\langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{global}}$ (Soft)")
        plt.ylabel(r"$\langle \mathrm{d}N_{D}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{D}/\mathrm{d}\eta \rangle_{\text{global}}$ (Hard)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="upper left", frameon=False, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(base_figuras_dir, f"Correlacao_dNdeta_eta_{corte_str}.pdf"), bbox_inches="tight")
        plt.close()

# =============================================================================
# 4. PLOT COMPARATIVO - TODOS OS CORTES DE ETA
# =============================================================================
if dados_curvas_eta:
    log_print(f"\n{'='*120}\nGERANDO PLOT COMPARATIVO COM TODOS OS CORTES DE ETA (dNdeta)\n{'='*120}")
    
    plt.figure(figsize=(8, 7))
    global_max_x, global_max_y = 0.0, 0.0

    for corte_eta in cortes_eta:
        if corte_eta not in dados_curvas_eta:
            continue
            
        x_norm, y_norm = dados_curvas_eta[corte_eta]
        estilo = estilos_eta[corte_eta]
        
        global_max_x = max(global_max_x, np.nanmax(x_norm))
        global_max_y = max(global_max_y, np.nanmax(y_norm))

        plt.plot(x_norm, y_norm, color=estilo["cor"], marker=estilo["marker"], 
                 linestyle='-', linewidth=1.5, markersize=7, alpha=0.85, 
                 label=rf"$|\eta| < {corte_eta}$")

    lim_x = global_max_x * 1.1
    lim_y = global_max_y * 1.1
    lim_ref = max(lim_x, lim_y)

    plt.plot([0, lim_ref], [0, lim_ref], 'k--', alpha=0.3, label="Linear scaling")
    plt.xlim(0, lim_x)
    plt.ylim(0, lim_y)

    plt.xlabel(r"$\langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{global}}$ (Soft)")
    plt.ylabel(r"$\langle \mathrm{d}N_{D}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{D}/\mathrm{d}\eta \rangle_{\text{global}}$ (Hard)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", frameon=False, title="All Physics", title_fontsize=11, fontsize=10)
    plt.tight_layout()
    
    caminho_fig_todos = os.path.join(base_figuras_dir, "Correlacao_dNdeta_todos_eta.pdf")
    plt.savefig(caminho_fig_todos, bbox_inches="tight")
    plt.close()
    
    log_print(f"[OK] Gráfico comparativo salvo em: {caminho_fig_todos}")

log_print(f"\n{'='*120}\nSucesso! Todos os plots de dN/deta foram salvos em: {base_figuras_dir}\n{'='*120}")
log_file.close()