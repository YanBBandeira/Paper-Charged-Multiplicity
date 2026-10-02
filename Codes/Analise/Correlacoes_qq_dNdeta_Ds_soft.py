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
    },
    "No Extra": {
        "pasta": "noExtra",
        "pref_soft": "Soft_14TeV_noExtra",
    },
    "No MPI": {
        "pasta": "noMPI",
        "pref_soft": "Soft_14TeV_noMPI",
    },
    "Only FSR": {
        "pasta": "onlyFSR",
        "pref_soft": "Soft_14TeV_onlyFSR",
    },
    "Only ISR": {
        "pasta": "onlyISR",
        "pref_soft": "Soft_14TeV_onlyISR",
    },
    "Only MPI": {
        "pasta": "onlyMPI",
        "pref_soft": "Soft_14TeV_onlyMPI",
    },
}

# Faixas diferenciais de momento transverso (pT_min, pT_max, identificador_para_arquivo, legenda)
faixas_pt = [
    (1.0,    2.0, "pt_1_2",     r"$1.0 < p_T < 2.0$ GeV/$c$"),
    (2.0,    4.0, "pt_2_4",     r"$2.0 < p_T < 4.0$ GeV/$c$"),
    (4.0,    6.0, "pt_4_6",     r"$4.0 < p_T < 6.0$ GeV/$c$"),
    (6.0,    8.0, "pt_6_8",     r"$6.0 < p_T < 8.0$ GeV/$c$"),
    (8.0,   12.0, "pt_8_12",    r"$8.0 < p_T < 12.0$ GeV/$c$"),
    (12.0,  24.0, "pt_12_24",   r"$12.0 < p_T < 24.0$ GeV/$c$")
]

# Configuração de cores e marcadores
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
log_file_path = os.path.join(diretorio_script, "validacao_correlacoes_ds_todos_casos_pt_SOFT_dndeta.log")
log_file = open(log_file_path, "w", encoding="utf-8")

def log_print(texto):
    print(texto)
    log_file.write(texto + "\n")


def escrever_tabela_latex(nome_caso, corte_eta, faixa_pt, dndeta_ch, dndeta_ds):
    """Acrescenta ao log uma tabela LaTeX com medias por classe de percentil baseada em dN/deta."""
    n_eventos_soft = len(dndeta_ch)
    if n_eventos_soft == 0:
        return

    dndeta_ch = np.asarray(dndeta_ch)
    dndeta_ds = np.asarray(dndeta_ds[:n_eventos_soft])
    ordem_soft = np.argsort(-dndeta_ch)
    eventos_classificados_soft = 0
    eventos_classificados_hard = 0
    soma_media_ponderada_soft = 0.0
    soma_media_ponderada_ds = 0.0

    def soma_probabilidades(valores):
        if len(valores) == 0 or not np.all(np.isfinite(valores)):
            return np.nan
        contagens = np.unique(valores, return_counts=True)[1]
        return np.sum(contagens / len(valores))

    soma_pn_ch_soft = soma_probabilidades(dndeta_ch)
    soma_pn_ds_soft = soma_probabilidades(dndeta_ds)

    log_file.write(
        f"\n% Caso: {nome_caso} | |eta| < {corte_eta} | pT: {faixa_pt}\n"
        "\\begin{tabular}{lccc}\n"
        "\\hline\n"
        "Class of Events & Percentile (\\%) & Média de dNch/deta & Média de dNDs/deta \\\\\n"
        "\\hline\n"
    )

    for nome_classe, percentual_inicial, percentual_final in classes_alice:
        inicio_soft = int(np.round(percentual_inicial / 100.0 * n_eventos_soft))
        fim_soft = max(inicio_soft + 1, int(np.round(percentual_final / 100.0 * n_eventos_soft)))
        classe_soft = dndeta_ch[ordem_soft][inicio_soft:fim_soft]
        classe_ds = dndeta_ds[ordem_soft][inicio_soft:fim_soft]
        media_carregadas = np.mean(classe_soft) if len(classe_soft) else 0.0
        media_ds = np.mean(classe_ds) if len(classe_ds) else 0.0
        eventos_classificados_soft += len(classe_soft)
        eventos_classificados_hard += len(classe_ds)
        soma_media_ponderada_soft += np.sum(classe_soft)
        soma_media_ponderada_ds += np.sum(classe_ds)
        percentual_inicial_fmt = f"{percentual_inicial:.2f}".rstrip("0").rstrip(".")
        percentual_final_fmt = f"{percentual_final:.2f}".rstrip("0").rstrip(".")
        if "." not in percentual_inicial_fmt:
            percentual_inicial_fmt += ".0"
        if "." not in percentual_final_fmt:
            percentual_final_fmt += ".0"
        log_file.write(
            f"{nome_classe} & ${percentual_inicial_fmt} - {percentual_final_fmt}$ "
            f"& {media_carregadas:.4f} & {media_ds:.6f} \\\\\n"
        )

    log_file.write("\\hline\n\\end{tabular}\n")
    cobertura_soft = eventos_classificados_soft / n_eventos_soft
    media_reconstruida_soft = soma_media_ponderada_soft / eventos_classificados_soft
    media_reconstruida_ds = soma_media_ponderada_ds / eventos_classificados_soft
    normalizacao_ok = all(
        np.isclose(valor, 1.0, atol=1e-12)
        for valor in (soma_pn_ch_soft, soma_pn_ds_soft)
    )
    cobertura_ok = np.isclose(cobertura_soft, 1.0)
    medias_ok = np.isclose(media_reconstruida_soft, np.mean(dndeta_ch)) and np.isclose(
        media_reconstruida_ds, np.mean(dndeta_ds)
    )
    status_sanidade = "OK" if normalizacao_ok and cobertura_ok and medias_ok else "ERRO"
    log_print(
        f"[Sanidade dN/deta | {nome_caso} | |eta| < {corte_eta} | {faixa_pt}]: "
        f"Cobertura classes = {cobertura_soft:.2%} | Status: {status_sanidade}"
    )
    log_file.flush()

# =============================================================================
# 2. FUNÇÕES DE CÁLCULO DE MÉDIAS LOCAIS POR PERCENTIL (dN/deta)
# =============================================================================
def obter_norm_dndeta(dndeta_ch, dndeta_ds, classes_info):
    n_evts = len(dndeta_ch)
    if n_evts == 0:
        return None, 0.0, None, 0.0

    media_glob = np.mean(dndeta_ch)
    media_glob_ds = np.mean(dndeta_ds)
    
    sort_idx = np.argsort(-dndeta_ch)
    ch_sorted = dndeta_ch[sort_idx]
    ds_sorted = dndeta_ds[sort_idx]

    x_norm_list = []
    y_norm_list = []
    
    for nome_cls, p_inf_pct, p_sup_pct in classes_info:
        i_start = int(np.round((p_inf_pct / 100.0) * n_evts))
        i_end = max(i_start + 1, int(np.round((p_sup_pct / 100.0) * n_evts)))

        faixa_ch = ch_sorted[i_start:i_end]
        m_local_ch = np.mean(faixa_ch) if len(faixa_ch) > 0 else 0.0
        norm_ch = m_local_ch / media_glob if media_glob > 0 else 0.0
        x_norm_list.append(norm_ch)

        faixa_ds = ds_sorted[i_start:i_end]
        m_local_ds = np.mean(faixa_ds) if len(faixa_ds) > 0 else 0.0
        norm_ds = m_local_ds / media_glob_ds if media_glob_ds > 0 else 0.0
        y_norm_list.append(norm_ds)
        
    return np.array(x_norm_list), media_glob, np.array(y_norm_list), media_glob_ds


# Estruturas globais para armazenar os dados por corte de eta e por faixa de pt
dados_global_por_eta = {corte: {} for corte in cortes_eta}
dados_global_por_pt = {pt_label: {corte: {} for corte in cortes_eta} for _, _, pt_label, _ in faixas_pt}

# =============================================================================
# 3. PROCESSAMENTO DOS CASOS, CORTES DE ETA E FAIXAS DE PT
# =============================================================================
for nome_caso, config in casos.items():
    log_print(f"\n{'#'*120}\nINICIANDO PROCESSAMENTO DO CASO: {nome_caso}\n{'#'*120}")
    
    caso_fig_dir = os.path.join(base_figuras_dir, config["pasta"])
    os.makedirs(caso_fig_dir, exist_ok=True)
    
    for corte_eta in cortes_eta:
        corte_str = f"{corte_eta:.1f}".replace(".", "p")
        delta_eta = 2.0 * corte_eta  # Largura da janela de pseudorapidez
        
        log_print(f"\n{'-'*80}\nPROCESSANDO {nome_caso} | |eta| < {corte_eta} (Delta eta = {delta_eta})\n{'-'*80}")

        # ---------------------------------------------------------------------
        # 3.1 PROCESSAMENTO GLOBAL
        # ---------------------------------------------------------------------
        arq_soft = os.path.join(base_dados_dir, "Soft", config["pasta"], f"{config['pref_soft']}_events_eta_{corte_str}_global.dat")

        if os.path.exists(arq_soft):
            dados_soft = np.loadtxt(arq_soft, skiprows=1)
           
            # Conversão para densidade dN/deta dividindo pela largura da janela
            dndeta_ch = dados_soft[:, 0] / delta_eta
            dndeta_ds = dados_soft[:, 3] / delta_eta  

            x_norm, _, y_norm, _ = obter_norm_dndeta(dndeta_ch, dndeta_ds, classes_alice)

            if x_norm is not None and y_norm is not None:
                dados_global_por_eta[corte_eta][nome_caso] = (x_norm, y_norm)
                escrever_tabela_latex(
                    nome_caso,
                    corte_eta,
                    "global (pT > 0)",
                    dndeta_ch,
                    dndeta_ds,
                )

        # ---------------------------------------------------------------------
        # 3.2 PROCESSAMENTO DIFERENCIAL POR FAIXA DE PT
        # ---------------------------------------------------------------------
        for pt_min, pt_max, pt_label, pt_desc in faixas_pt:
            arq_soft_pt = os.path.join(base_dados_dir, "Soft", config["pasta"], f"{config['pref_soft']}_events_eta_{corte_str}_{pt_label}.dat")
            
            if not os.path.exists(arq_soft_pt):
                continue

            dados_soft_pt = np.loadtxt(arq_soft_pt, skiprows=1)

            dndeta_ch_pt = dados_soft_pt[:, 0] / delta_eta
            dndeta_ds_pt = dados_soft_pt[:, 3] / delta_eta

            x_norm_pt, _, y_norm_pt, _ = obter_norm_dndeta(dndeta_ch_pt, dndeta_ds_pt, classes_alice)

            if x_norm_pt is not None and y_norm_pt is not None:
                dados_global_por_pt[pt_label][corte_eta][nome_caso] = (x_norm_pt, y_norm_pt)
                escrever_tabela_latex(
                    nome_caso,
                    corte_eta,
                    f"{pt_min:g} - {pt_max:g} GeV/c",
                    dndeta_ch_pt,
                    dndeta_ds_pt
                )

                pt_fig_dir = os.path.join(caso_fig_dir, "Por_Pt")
                os.makedirs(pt_fig_dir, exist_ok=True)

# =============================================================================
# 4. GERAÇÃO DE FIGURAS GLOBAIS E DIFERENCIAIS (COMPARATIVO DE ETA)
# =============================================================================
log_print(f"\n{'#'*120}\nGERANDO FIGURAS GLOBAIS (COMPARATIVO DE ETA)\n{'#'*120}")

for nome_caso, config in casos.items():
    caso_fig_dir = os.path.join(base_figuras_dir, config["pasta"], "Por_Pt")
    os.makedirs(caso_fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 7))
    global_max_x, global_max_y = 0.0, 0.0
    tem_dados = False

    for corte_eta in cortes_eta:
        if nome_caso not in dados_global_por_eta[corte_eta]:
            continue

        x_norm, y_norm = dados_global_por_eta[corte_eta][nome_caso]
        tem_dados = True
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
            label=rf"$|\eta| < {corte_eta}$",
        )

    if tem_dados:
        lim_x = global_max_x * 1.1
        lim_y = global_max_y * 1.1
        lim_ref = max(lim_x, lim_y)
        plt.plot([0, lim_ref], [0, lim_ref], "k--", alpha=0.3, label="Linear scaling")
        plt.xlim(0, lim_x)
        plt.ylim(0, lim_y)
        plt.xlabel(r"$\langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{global}}$")
        plt.ylabel(r"$\langle \mathrm{d}N_{D_s}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{D_s}/\mathrm{d}\eta \rangle_{\text{global}}$ ($D_s$ Mesons)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="best", frameon=False, title=f"{nome_caso}\nGlobal ($p_T > 0$)", title_fontsize=10, fontsize=9)
        plt.tight_layout()

        caminho_fig_global = os.path.join(caso_fig_dir, "Correlacao_QQ_Ds_pt_global_dndeta.pdf")
        plt.savefig(caminho_fig_global, bbox_inches="tight")
        plt.close()
        log_print(f"[OK] Figura global salva em: {caminho_fig_global}")

log_print(f"\n{'#'*120}\nGERANDO FIGURAS DIFERENCIAIS POR FAIXA DE PT (COMPARATIVO DE ETA)\n{'#'*120}")

for nome_caso, config in casos.items():
    caso_fig_dir = os.path.join(base_figuras_dir, config["pasta"], "Por_Pt")
    os.makedirs(caso_fig_dir, exist_ok=True)

    for pt_min, pt_max, pt_label, pt_desc in faixas_pt:
        plt.figure(figsize=(8, 7))
        global_max_x, global_max_y = 0.0, 0.0
        tem_dados = False

        for corte_eta in cortes_eta:
            corte_str = f"{corte_eta:.1f}".replace(".", "p")
            delta_eta = 2.0 * corte_eta
            arq_soft_pt = os.path.join(base_dados_dir, "Soft", config["pasta"], f"{config['pref_soft']}_events_eta_{corte_str}_{pt_label}.dat")

            if not os.path.exists(arq_soft_pt):
                continue

            dados_soft_pt = np.loadtxt(arq_soft_pt, skiprows=1)

            dndeta_ch_pt = dados_soft_pt[:, 0] / delta_eta
            dndeta_ds_pt = dados_soft_pt[:, 3] / delta_eta

            x_norm, _, y_norm, _ = obter_norm_dndeta(dndeta_ch_pt, dndeta_ds_pt, classes_alice)

            if x_norm is not None and y_norm is not None:
                tem_dados = True
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

        if tem_dados:
            lim_x = global_max_x * 1.1
            lim_y = global_max_y * 1.1
            lim_ref = max(lim_x, lim_y)

            plt.plot([0, lim_ref], [0, lim_ref], "k--", alpha=0.3, label="Linear scaling")
            plt.xlim(0, lim_x)
            plt.ylim(0, lim_y)

            plt.xlabel(r"$\langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{global}}$")
            plt.ylabel(r"$\langle \mathrm{d}N_{D_s}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{D_s}/\mathrm{d}\eta \rangle_{\text{global}}$ ($D_s$ Mesons)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(loc="best", frameon=False, title=f"{nome_caso}\n{pt_desc}", title_fontsize=10, fontsize=9)
            plt.tight_layout()

            caminho_fig_pt = os.path.join(caso_fig_dir, f"Correlacao_QQ_Ds_{pt_label}_dndeta.pdf")
            plt.savefig(caminho_fig_pt, bbox_inches="tight")
            plt.close()
            log_print(f"[OK] Figura de pT salva em: {caminho_fig_pt}")

# =============================================================================
# 5. COMPARATIVO ENTRE OS CASOS POR FAIXA DE PT E CORTE DE ETA
# =============================================================================
log_print(f"\n{'#'*120}\nGERANDO COMPARATIVOS ENTRE CASOS POR FAIXA DE PT\n{'#'*120}")

figuras_casos_pt_dir = os.path.join(base_figuras_dir, "Comparativo_Casos_Por_Pt")
os.makedirs(figuras_casos_pt_dir, exist_ok=True)

for corte_eta in cortes_eta:
    corte_str = f"{corte_eta:.1f}".replace(".", "p")
    plt.figure(figsize=(8, 7))
    global_max_x, global_max_y = 0.0, 0.0
    tem_dados_caso = False

    for nome_caso in casos:
        if nome_caso not in dados_global_por_eta[corte_eta]:
            continue

        tem_dados_caso = True
        x_norm, y_norm = dados_global_por_eta[corte_eta][nome_caso]
        estilo = estilos_casos[nome_caso]
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
            label=nome_caso,
        )

    if tem_dados_caso:
        lim_x = global_max_x * 1.1
        lim_y = global_max_y * 1.1
        lim_ref = max(lim_x, lim_y)
        plt.plot([0, lim_ref], [0, lim_ref], "k--", alpha=0.3, label="Linear scaling")
        plt.xlim(0, lim_x)
        plt.ylim(0, lim_y)
        plt.xlabel(r"$\langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{global}}$")
        plt.ylabel(r"$\langle \mathrm{d}N_{D_s}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{D_s}/\mathrm{d}\eta \rangle_{\text{global}}$ ($D_s$ Mesons)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="best", frameon=False, title=rf"Casos físicos\n$|\eta| < {corte_eta}$ | Global ($p_T > 0$)", title_fontsize=10, fontsize=9)
        plt.tight_layout()

        caminho_fig_global_casos = os.path.join(
            figuras_casos_pt_dir, f"Comparativo_Casos_eta_{corte_str}_pt_global_dndeta.pdf"
        )
        plt.savefig(caminho_fig_global_casos, bbox_inches="tight")
        plt.close()
        log_print(f"[OK] Comparativo global para $|\eta| < {corte_eta}$ salvo em: {caminho_fig_global_casos}")

for pt_min, pt_max, pt_label, pt_desc in faixas_pt:
    for corte_eta in cortes_eta:
        corte_str = f"{corte_eta:.1f}".replace(".", "p")
        
        plt.figure(figsize=(8, 7))
        global_max_x, global_max_y = 0.0, 0.0
        tem_dados_caso = False

        for nome_caso in casos.keys():
            if nome_caso in dados_global_por_pt[pt_label][corte_eta]:
                tem_dados_caso = True
                x_norm, y_norm = dados_global_por_pt[pt_label][corte_eta][nome_caso]
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

        if tem_dados_caso:
            lim_x = global_max_x * 1.1
            lim_y = global_max_y * 1.1
            lim_ref = max(lim_x, lim_y)

            plt.plot([0, lim_ref], [0, lim_ref], "k--", alpha=0.3, label="Linear scaling")
            plt.xlim(0, lim_x)
            plt.ylim(0, lim_y)

            plt.xlabel(r"$\langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{\text{ch}}/\mathrm{d}\eta \rangle_{\text{global}}$")
            plt.ylabel(r"$\langle \mathrm{d}N_{D_s}/\mathrm{d}\eta \rangle_{\text{local}} / \langle \mathrm{d}N_{D_s}/\mathrm{d}\eta \rangle_{\text{global}}$ ($D_s$ Mesons)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(loc="best", frameon=False, title=f"Casos Físicos\n$|\eta| < {corte_eta}$ | {pt_desc}", title_fontsize=10, fontsize=9)
            plt.tight_layout()

            caminho_fig_caso_pt = os.path.join(figuras_casos_pt_dir, f"Comparativo_Casos_eta_{corte_str}_{pt_label}_dndeta.pdf")
            plt.savefig(caminho_fig_caso_pt, bbox_inches="tight")
            plt.close()
            log_print(f"[OK] Comparativo de casos para $|\eta| < {corte_eta}$ e {pt_label} salvo em: {caminho_fig_caso_pt}")

log_print(f"\n{'#'*120}\nSucesso! Todas as análises em dN/deta foram processadas e geradas.\n{'#'*120}")
log_file.close()