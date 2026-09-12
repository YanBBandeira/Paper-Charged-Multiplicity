import os
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use(hep.style.ROOT)

# =============================================================================
# 1. CLASSES DE MULTIPLICIDADE OFICIAIS ALICE (CLASSES I A X)
# =============================================================================
classes_alice = [
    ("I", 0.0, 0.95),
    ("II", 0.95, 4.7),
    ("III", 4.7, 9.5),
    ("IV", 9.5, 14.0),
    ("V", 14.0, 19.0),
    ("VI", 19.0, 28.0),
    ("VII", 28.0, 38.0),
    ("VIII", 38.0, 48.0),
    ("IX", 48.0, 68.0),
    ("X", 68.0, 100.0),
]

# Todos os 6 casos de simulação físicos mapeados corretamente
casos = {
    "All Physics": {
        "pasta": "all",
        "pref_soft": "Soft_14TeV_all",
        "pref_hard": "Hard_14TeV_all",
    },
    "No Extra": {
        "pasta": "noExtra",
        "pref_soft": "Soft_14TeV_noExtra",
        "pref_hard": "Hard_14TeV_noExtra",
    },
    "No MPI": {
        "pasta": "noMPI",
        "pref_soft": "Soft_14TeV_noMPI",
        "pref_hard": "Hard_14TeV_noMPI",
    },
    "Only FSR": {
        "pasta": "onlyFSR",
        "pref_soft": "Soft_14TeV_onlyFSR",
        "pref_hard": "Hard_14TeV_onlyFSR",
    },
    "Only ISR": {
        "pasta": "onlyISR",
        "pref_soft": "Soft_14TeV_onlyISR",
        "pref_hard": "Hard_14TeV_onlyISR",
    },
    "Only MPI": {
        "pasta": "onlyMPI",
        "pref_soft": "Soft_14TeV_onlyMPI",
        "pref_hard": "Hard_14TeV_onlyMPI",
    },
}

# Configuração de cores e marcadores para o comparativo de cortes em eta
estilos_eta = {
    0.5: {"cor": "black", "marker": "o"},
    1.0: {"cor": "firebrick", "marker": "s"},
    2.0: {"cor": "royalblue", "marker": "^"},
    3.0: {"cor": "forestgreen", "marker": "v"},
    4.0: {"cor": "darkorange", "marker": "D"},
    5.0: {"cor": "purple", "marker": "p"},
}

estilos_casos = {
    "All Physics": {"cor": "black", "marker": "o"},
    "No Extra": {"cor": "firebrick", "marker": "s"},
    "No MPI": {"cor": "royalblue", "marker": "^"},
    "Only FSR": {"cor": "forestgreen", "marker": "v"},
    "Only ISR": {"cor": "darkorange", "marker": "D"},
    "Only MPI": {"cor": "purple", "marker": "p"},
}

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

diretorio_script = os.path.dirname(os.path.abspath(__file__))
base_dados_dir = os.path.abspath(os.path.join(diretorio_script, "..", "Dados"))
base_figuras_dir = os.path.join(diretorio_script, "Figures", "Correlacoes_Ds")
figuras_comparativo_casos_dir = os.path.join(base_figuras_dir, "Comparativo_Casos_Por_Eta")

os.makedirs(base_figuras_dir, exist_ok=True)
os.makedirs(figuras_comparativo_casos_dir, exist_ok=True)
log_file_path = os.path.join(diretorio_script, "validacao_correlacoes_ds_todos_casos.log")
log_file = open(log_file_path, "w", encoding="utf-8")

def log_print(texto):
    print(texto)
    log_file.write(texto + "\n")

# =============================================================================
# 2. FUNÇÕES DE CÁLCULO DE MÉDIAS LOCAIS POR PERCENTIL
# =============================================================================
def obter_norm_soft(n_ch_soft, classes_info):
    n_evts = len(n_ch_soft)
    if n_evts == 0:
        return None, 0.0, []

    media_glob = np.mean(n_ch_soft)
    sort_idx = np.argsort(-n_ch_soft)
    n_ch_sorted = n_ch_soft[sort_idx]

    x_norm_list, log_soft = [], []
    for nome_cls, p_inf_pct, p_sup_pct in classes_info:
        i_start = int(np.round((p_inf_pct / 100.0) * n_evts))
        i_end = max(i_start + 1, int(np.round((p_sup_pct / 100.0) * n_evts)))

        faixa = n_ch_sorted[i_start:i_end]
        m_local = np.mean(faixa) if len(faixa) > 0 else 0.0
        norm = m_local / media_glob if media_glob > 0 else 0.0

        x_norm_list.append(norm)
        log_soft.append((nome_cls, np.min(faixa) if len(faixa) > 0 else 0, np.max(faixa) if len(faixa) > 0 else 0, m_local, norm))

    return np.array(x_norm_list), media_glob, log_soft

def obter_norm_hard(n_ch_hard, n_ds_hard, classes_info):
    n_evts = len(n_ch_hard)
    if n_evts == 0:
        return None, 0.0, []

    media_glob_ds = np.mean(n_ds_hard)
    sort_idx = np.argsort(-n_ch_hard)
    n_ch_sorted, n_ds_sorted = n_ch_hard[sort_idx], n_ds_hard[sort_idx]

    y_norm_list, log_hard = [], []
    for nome_cls, p_inf_pct, p_sup_pct in classes_info:
        i_start = int(np.round((p_inf_pct / 100.0) * n_evts))
        i_end = max(i_start + 1, int(np.round((p_sup_pct / 100.0) * n_evts)))

        faixa_ch, faixa_ds = n_ch_sorted[i_start:i_end], n_ds_sorted[i_start:i_end]
        m_local_ds = np.mean(faixa_ds) if len(faixa_ds) > 0 else 0.0
        norm_ds = m_local_ds / media_glob_ds if media_glob_ds > 0 else 0.0

        y_norm_list.append(norm_ds)
        log_hard.append((nome_cls, np.min(faixa_ch) if len(faixa_ch) > 0 else 0, np.max(faixa_ch) if len(faixa_ch) > 0 else 0, m_local_ds, norm_ds))

    return np.array(y_norm_list), media_glob_ds, log_hard


dados_global_por_eta = {corte: {} for corte in cortes_eta}

# =============================================================================
# 3. PROCESSAMENTO DOS CASOS E CORTES DE ETA
# =============================================================================
for nome_caso, config in casos.items():
    log_print(f"\n{'#'*120}\nINICIANDO PROCESSAMENTO DO CASO: {nome_caso}\n{'#'*120}")
    
    caso_fig_dir = os.path.join(base_figuras_dir, config["pasta"])
    os.makedirs(caso_fig_dir, exist_ok=True)
    
    dados_curvas_eta = {} 
    x_norm_ref_0p5 = None # Armazenará o eixo X fixado no corte eta < 0.5 para este caso

    for corte_eta in cortes_eta:
        corte_str = f"{corte_eta:.1f}".replace(".", "p")
        log_print(f"\n{'-'*80}\nPROCESSANDO {nome_caso} | |eta| < {corte_eta}\n{'-'*80}")

        arq_soft = os.path.join(base_dados_dir, "Soft", config["pasta"], f"{config['pref_soft']}_events_eta_{corte_str}.dat")
        arq_hard = os.path.join(base_dados_dir, "Hard", config["pasta"], f"{config['pref_hard']}_events_eta_{corte_str}.dat")

        if not os.path.exists(arq_soft) or not os.path.exists(arq_hard):
            log_print(f"[AVISO] Arquivos inexistentes para {nome_caso} em |eta| < {corte_eta}")
            continue

        dados_soft = np.loadtxt(arq_soft, skiprows=1)
        dados_hard = np.loadtxt(arq_hard, skiprows=1)

        n_ch_soft = dados_soft[:, 0] if dados_soft.ndim > 1 else dados_soft
        n_ch_hard, n_ds_hard = dados_hard[:, 0], dados_hard[:, 2]

        x_norm, media_ch_glob, log_soft = obter_norm_soft(n_ch_soft, classes_alice)
        y_norm, media_ds_glob, log_hard = obter_norm_hard(n_ch_hard, n_ds_hard, classes_alice)

        if x_norm is not None and y_norm is not None:
            dados_curvas_eta[corte_eta] = (x_norm, y_norm)
            
            # Guarda a referência do eixo X para eta < 0.5 caso precise fixar
            if corte_eta == 0.5:
                x_norm_ref_0p5 = x_norm

            # Salva na estrutura global para a terceira figura (comparativo entre casos)
            dados_global_por_eta[corte_eta][nome_caso] = (x_norm, y_norm)

            log_print(f"\n>>> {nome_caso} | |eta| < {corte_eta} | Evts Soft: {len(n_ch_soft)} | Evts Hard: {len(n_ch_hard)}")
            log_print(f"    <<N_ch>>_soft = {media_ch_glob:.4f} | <<N_Ds>>_hard = {media_ds_glob:.6f}")

    # =========================================================================
    # FIGURA 1: COMPARATIVO PADRÃO DE TODOS OS CORTES DE ETA (X e Y variando juntos)
    # =========================================================================
    if dados_curvas_eta:
        log_print(f"\nGERANDO FIGURA 1 (Comparativo de Eta) PARA {nome_caso}")

        plt.figure(figsize=(8, 7))
        global_max_x, global_max_y = 0.0, 0.0

        for corte_eta in cortes_eta:
            if corte_eta not in dados_curvas_eta: continue
            
            x_norm, y_norm = dados_curvas_eta[corte_eta]
            estilo = estilos_eta[corte_eta]

            global_max_x = max(global_max_x, np.nanmax(x_norm))
            global_max_y = max(global_max_y, np.nanmax(y_norm))

            plt.plot(
                x_norm, y_norm,
                color=estilo["cor"],
                marker=estilo["marker"],
                linestyle="-",
                linewidth=1.5,
                markersize=7,
                alpha=0.85,
                label=rf"$|\eta| < {corte_eta}$"
            )

        lim_x = global_max_x * 1.1
        lim_y = global_max_y * 1.1
        lim_ref = max(lim_x, lim_y)

        plt.plot([0, lim_ref], [0, lim_ref], "k--", alpha=0.3, label="Linear scaling")
        plt.xlim(0, lim_x)
        plt.ylim(0, lim_y)

        plt.xlabel(r"$\langle N_{\text{ch}} \rangle_{\text{local}} / \langle N_{\text{ch}} \rangle_{\text{global}}$ (Soft)")
        plt.ylabel(r"$\langle N_{D_s} \rangle_{\text{local}} / \langle N_{D_s} \rangle_{\text{global}}$ ($D_s$ Mesons)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="upper left", frameon=False, title=f"{nome_caso} ($D_s$)", title_fontsize=11, fontsize=10)
        plt.tight_layout()

        caminho_fig_todos = os.path.join(caso_fig_dir, "Correlacao_QQ_Ds_todos_eta.pdf")
        plt.savefig(caminho_fig_todos, bbox_inches="tight")
        plt.close()
        log_print(f"[OK] Figura 1 salva em: {caminho_fig_todos}")
    
    # =========================================================================
    # SEGUNDA FIGURA: COMPARATIVO DE ETA COM EIXO X FIXADO EM |eta| < 0.5
    # =========================================================================
    if dados_curvas_eta:
        log_print(f"\nGERANDO FIGURAS 2 (Eixo X fixado por corte) PARA {nome_caso}")

        for corte_ref in cortes_eta:
            if corte_ref not in dados_curvas_eta:
                continue

            corte_ref_str = f"{corte_ref:.1f}".replace(".", "p")
            plt.figure(figsize=(8, 7))
            global_max_x, global_max_y = 0.0, 0.0
            x_base = dados_curvas_eta[corte_ref][0] # Eixo X fixo baseado no corte de referência atual

            for corte_eta in cortes_eta:
                if corte_eta not in dados_curvas_eta: continue
                
                _, y_norm = dados_curvas_eta[corte_eta]
                estilo = estilos_eta[corte_eta]

                global_max_x = max(global_max_x, np.nanmax(x_base))
                global_max_y = max(global_max_y, np.nanmax(y_norm))

                plt.plot(
                    x_base, y_norm,
                    color=estilo["cor"],
                    marker=estilo["marker"],
                    linestyle="-",
                    linewidth=1.5,
                    markersize=7,
                    alpha=0.85,
                    label=rf"$|\eta| < {corte_eta}$ (Y) | Ref: $|\eta| < {corte_ref}$ (X)"
                )

            lim_x = global_max_x * 1.1
            lim_y = global_max_y * 1.1
            lim_ref_lim = max(lim_x, lim_y)

            plt.plot([0, lim_ref_lim], [0, lim_ref_lim], "k--", alpha=0.3, label="Linear scaling")
            plt.xlim(0, lim_x)
            plt.ylim(0, lim_y)

            plt.xlabel(rf"$\langle N_{{\text{{ch}}}} \rangle_{{\text{{local}}}} / \langle N_{{\text{{ch}}}} \rangle_{{\text{{global}}}}$ (Soft, $|\eta| < {corte_ref}$)")
            plt.ylabel(r"$\langle N_{D_s} \rangle_{\text{local}} / \langle N_{D_s} \rangle_{\text{global}}$ ($D_s$ Mesons)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(loc="upper left", frameon=False, title=f"{nome_caso} (X: $|\eta| < {corte_ref}$)", title_fontsize=11, fontsize=9)
            plt.tight_layout()

            caminho_fig_fixado = os.path.join(caso_fig_dir, f"Correlacao_QQ_Ds_X_fixado_eta_{corte_ref_str}.pdf")
            plt.savefig(caminho_fig_fixado, bbox_inches="tight")
            plt.close()
            log_print(f"[OK] Figura 2 com X fixado em $|\eta| < {corte_ref}$ salva em: {caminho_fig_fixado}")

# =============================================================================
# TERCEIRA FIGURA: COMPARATIVO ENTRE OS CASOS PARA UM MESMO CORTE EM ETA
# =============================================================================
log_print(f"\n{'#'*120}\nGERANDO TERCEIRA FIGURA: COMPARATIVO ENTRE CASOS POR CORTE DE ETA\n{'#'*120}")

for corte_eta in cortes_eta:
    if not dados_global_por_eta[corte_eta]:
        continue

    corte_str = f"{corte_eta:.1f}".replace(".", "p")
    plt.figure(figsize=(8, 7))
    global_max_x, global_max_y = 0.0, 0.0

    for nome_caso, (x_norm, y_norm) in dados_global_por_eta[corte_eta].items():
        estilo = estilos_casos.get(nome_caso, {"cor": "gray", "marker": "o"})

        global_max_x = max(global_max_x, np.nanmax(x_norm))
        global_max_y = max(global_max_y, np.nanmax(y_norm))

        plt.plot(
            x_norm, y_norm,
            color=estilo["cor"],
            marker=estilo["marker"],
            linestyle="-",
            linewidth=1.5,
            markersize=7,
            alpha=0.85,
            label=nome_caso
        )

    lim_x = global_max_x * 1.1
    lim_y = global_max_y * 1.1
    lim_ref = max(lim_x, lim_y)

    plt.plot([0, lim_ref], [0, lim_ref], "k--", alpha=0.3, label="Linear scaling")
    plt.xlim(0, lim_x)
    plt.ylim(0, lim_y)

    plt.xlabel(r"$\langle N_{\text{ch}} \rangle_{\text{local}} / \langle N_{\text{ch}} \rangle_{\text{global}}$ (Soft)")
    plt.ylabel(r"$\langle N_{D_s} \rangle_{\text{local}} / \langle N_{D_s} \rangle_{\text{global}}$ ($D_s$ Mesons)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", frameon=False, title=f"Casos Físicos ($|\eta| < {corte_eta}$)", title_fontsize=11, fontsize=9)
    plt.tight_layout()

    caminho_fig_casos = os.path.join(figuras_comparativo_casos_dir, f"Comparativo_Casos_eta_{corte_str}.pdf")
    plt.savefig(caminho_fig_casos, bbox_inches="tight")
    plt.close()

    log_print(f"[OK] Comparativo entre casos para $|\eta| < {corte_eta}$ salvo em: {caminho_fig_casos}")

log_print(f"\n{'#'*120}\nSucesso! Todas as modificações e novas figuras foram geradas.\n{'#'*120}")
log_file.close()