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

casos = {
    "All Physics": {
        "pasta": "all",
        "pref_soft": "Soft_14TeV_all",
        "pref_hard": "Hard_14TeV_all",
    }
}

# Configuração de cores e marcadores para o gráfico comparativo dos cortes em eta
estilos_eta = {
    0.5: {"cor": "black", "marker": "o"},
    1.0: {"cor": "firebrick", "marker": "s"},
    2.0: {"cor": "royalblue", "marker": "^"},
    3.0: {"cor": "forestgreen", "marker": "v"},
    4.0: {"cor": "darkorange", "marker": "D"},
    5.0: {"cor": "purple", "marker": "p"},
}

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

diretorio_script = os.path.dirname(os.path.abspath(__file__))
base_dados_dir = os.path.abspath(
    os.path.join(diretorio_script, "..", "Dados")
)
base_figuras_dir = os.path.join(diretorio_script, "Figures", "Correlacoes_All")

os.makedirs(base_figuras_dir, exist_ok=True)
log_file_path = os.path.join(diretorio_script, "validacao_correlacoes_all.log")
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
    log_soft.append((nome_cls, np.min(faixa), np.max(faixa), m_local, norm))

  return np.array(x_norm_list), media_glob, log_soft


def obter_norm_hard(n_ch_hard, n_d_hard, classes_info):
  n_evts = len(n_ch_hard)
  if n_evts == 0:
    return None, 0.0, []

  media_glob_d = np.mean(n_d_hard)
  sort_idx = np.argsort(-n_ch_hard)
  n_ch_sorted, n_d_sorted = n_ch_hard[sort_idx], n_d_hard[sort_idx]

  y_norm_list, log_hard = [], []
  for nome_cls, p_inf_pct, p_sup_pct in classes_info:
    i_start = int(np.round((p_inf_pct / 100.0) * n_evts))
    i_end = max(i_start + 1, int(np.round((p_sup_pct / 100.0) * n_evts)))

    faixa_ch, faixa_d = n_ch_sorted[i_start:i_end], n_d_sorted[i_start:i_end]
    m_local_d = np.mean(faixa_d) if len(faixa_d) > 0 else 0.0
    norm_d = m_local_d / media_glob_d if media_glob_d > 0 else 0.0

    y_norm_list.append(norm_d)
    log_hard.append(
        (nome_cls, np.min(faixa_ch), np.max(faixa_ch), m_local_d, norm_d)
    )

  return np.array(y_norm_list), media_glob_d, log_hard


# =============================================================================
# 3. PROCESSAMENTO - PLOTS INDIVIDUAIS E ARMAZENAMENTO DAS CURVAS
# =============================================================================
dados_curvas_eta = {}  # Guarda (x_norm, y_norm) de cada eta

for corte_eta in cortes_eta:
  corte_str = f"{corte_eta:.1f}".replace(".", "p")
  log_print(
      f"\n{'='*120}\nPROCESSANDO |eta| < {corte_eta} (CASO ALL"
      f" PHYSICS)\n{'='*120}"
  )

  config = casos["All Physics"]
  arq_soft = os.path.join(
      base_dados_dir,
      "Soft",
      config["pasta"],
      f"{config['pref_soft']}_events_eta_{corte_str}.dat",
  )
  arq_hard = os.path.join(
      base_dados_dir,
      "Hard",
      config["pasta"],
      f"{config['pref_hard']}_events_eta_{corte_str}.dat",
  )

  if not os.path.exists(arq_soft) or not os.path.exists(arq_hard):
    log_print(
        f"[AVISO] Arquivos inexistentes para All Physics em |eta| < {corte_eta}"
    )
    continue

  dados_soft = np.loadtxt(arq_soft, skiprows=1)
  dados_hard = np.loadtxt(arq_hard, skiprows=1)

  n_ch_soft = dados_soft[:, 0] if dados_soft.ndim > 1 else dados_soft
  n_ch_hard, n_d_hard = dados_hard[:, 0], dados_hard[:, 1]

  x_norm, media_ch_glob, log_soft = obter_norm_soft(n_ch_soft, classes_alice)
  y_norm, media_d_glob, log_hard = obter_norm_hard(
      n_ch_hard, n_d_hard, classes_alice
  )

  if x_norm is not None and y_norm is not None:
    dados_curvas_eta[corte_eta] = (x_norm, y_norm)

    log_print(
        f"\n>>> CASO: All Physics | Evts Soft: {len(n_ch_soft)} | Evts Hard:"
        f" {len(n_ch_hard)}"
    )
    log_print(
        f"    <<N_ch>>_soft = {media_ch_glob:.4f} | <<N_D>>_hard ="
        f" {media_d_glob:.6f}"
    )
    log_print(
        f"   {'Classe':<6} | {'N_ch_soft (Faixa)':<18} | {'<N_ch>_soft':<12} |"
        f" {'X_norm (Soft)':<15} | {'N_ch_hard (Faixa)':<18} |"
        f" {'<N_D>_hard':<12} | {'Y_norm (Hard)':<15}"
    )
    log_print("   " + "-" * 115)

    for i in range(len(classes_alice)):
      c_nome = classes_alice[i][0]
      faixa_s = f"[{log_soft[i][1]:.0f}, {log_soft[i][2]:.0f}]"
      faixa_h = f"[{log_hard[i][1]:.0f}, {log_hard[i][2]:.0f}]"
      log_print(
          f"   {c_nome:<6} | {faixa_s:<18} | {log_soft[i][3]:<12.2f} |"
          f" {log_soft[i][4]:<15.4f} | {faixa_h:<18} | {log_hard[i][3]:<12.4f} |"
          f" {log_hard[i][4]:<15.4f}"
      )

    # Plot Individual
    plt.figure(figsize=(8, 7))
    estilo = estilos_eta[corte_eta]
    plt.plot(
        x_norm,
        y_norm,
        color=estilo["cor"],
        marker=estilo["marker"],
        linestyle="-",
        linewidth=1.5,
        markersize=7,
        alpha=0.9,
        label=f"All Physics ($|\\eta| < {corte_eta}$)",
    )

    lim_x = np.nanmax(x_norm) * 1.1
    lim_y = np.nanmax(y_norm) * 1.1
    lim_ref = max(lim_x, lim_y)

    plt.plot([0, lim_ref], [0, lim_ref], "k--", alpha=0.3, label="Linear scaling")
    plt.xlim(0, lim_x)
    plt.ylim(0, lim_y)

    plt.xlabel(
        r"$\langle N_{\text{ch}} \rangle_{\text{local}} / \langle"
        r" N_{\text{ch}} \rangle_{\text{global}}$ (Soft)"
    )
    plt.ylabel(
        r"$\langle N_{D} \rangle_{\text{local}} / \langle N_{D}"
        r" \rangle_{\text{global}}$ (Hard)"
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(base_figuras_dir, f"Correlacao_QQ_eta_{corte_str}.pdf"),
        bbox_inches="tight",
    )
    plt.close()

# =============================================================================
# 4. PLOT COMPARATIVO - 6 CURVAS DE ETA (EIXO X VARIÁVEL COM ETA)
# =============================================================================
if dados_curvas_eta:
  log_print(
      f"\n{'='*120}\nGERANDO PLOT COMPARATIVO COM TODOS OS CORTES DE"
      f" ETA\n{'='*120}"
  )

  plt.figure(figsize=(8, 7))
  global_max_x, global_max_y = 0.0, 0.0

  for corte_eta in cortes_eta:
    if corte_eta not in dados_curvas_eta:
      continue

    x_norm, y_norm = dados_curvas_eta[corte_eta]
    estilo = estilos_eta[corte_eta]

    global_max_x = max(global_max_x, np.nanmax(x_norm))
    global_max_y = max(global_max_y, np.nanmax(y_norm))

    plt.plot(
        x_norm,
        y_norm,
        color=estilo["cor"],
        marker=estilo["marker"],
        linestyle="-",
        linewidth=1.5,
        markersize=7,
        alpha=0.85,
        label=rf"$|\eta| < {corte_eta}$",
    )

  lim_x = global_max_x * 1.1
  lim_y = global_max_y * 1.1
  lim_ref = max(lim_x, lim_y)

  plt.plot([0, lim_ref], [0, lim_ref], "k--", alpha=0.3, label="Linear scaling")
  plt.xlim(0, lim_x)
  plt.ylim(0, lim_y)

  plt.xlabel(
      r"$\langle N_{\text{ch}} \rangle_{\text{local}} / \langle"
      r" N_{\text{ch}} \rangle_{\text{global}}$ (Soft)"
  )
  plt.ylabel(
      r"$\langle N_{D} \rangle_{\text{local}} / \langle N_{D}"
      r" \rangle_{\text{global}}$ (Hard)"
  )
  plt.grid(True, linestyle="--", alpha=0.7)
  plt.legend(
      loc="upper left",
      frameon=False,
      title="All Physics",
      title_fontsize=11,
      fontsize=10,
  )
  plt.tight_layout()

  caminho_fig_todos = os.path.join(
      base_figuras_dir, "Correlacao_QQ_todos_eta.pdf"
  )
  plt.savefig(caminho_fig_todos, bbox_inches="tight")
  plt.close()

  log_print(f"[OK] Gráfico comparativo salvo em: {caminho_fig_todos}")

# =============================================================================
# 5. PLOT COMPARATIVO - EIXO X FIXO EM |\eta| < 0.5
# =============================================================================
if 0.5 in dados_curvas_eta:
  log_print(
      f"\n{'='*120}\nGERANDO PLOT COMPARATIVO COM EIXO X FIXO EM |\eta| <"
      f" 0.5\n{'='*120}"
  )

  x_norm_ref = dados_curvas_eta[0.5][0]  # Pega o eixo X de |eta| < 0.5

  plt.figure(figsize=(8, 7))
  global_max_x = np.nanmax(x_norm_ref)
  global_max_y = 0.0

  for corte_eta in cortes_eta:
    if corte_eta not in dados_curvas_eta:
      continue

    _, y_norm = dados_curvas_eta[corte_eta]
    estilo = estilos_eta[corte_eta]

    global_max_y = max(global_max_y, np.nanmax(y_norm))

    plt.plot(
        x_norm_ref,
        y_norm,
        color=estilo["cor"],
        marker=estilo["marker"],
        linestyle="-",
        linewidth=1.5,
        markersize=7,
        alpha=0.85,
        label=rf"$|\eta|_{{\text{{Hard}}}} < {corte_eta}$",
    )

  lim_x = global_max_x * 1.1
  lim_y = global_max_y * 1.1
  lim_ref = max(lim_x, lim_y)

  plt.plot([0, lim_ref], [0, lim_ref], "k--", alpha=0.3, label="Linear scaling")
  plt.xlim(0, lim_x)
  plt.ylim(0, lim_y)

  plt.xlabel(
      r"$\langle N_{\text{ch}} \rangle_{\text{local}} / \langle"
      r" N_{\text{ch}} \rangle_{\text{global}} \quad (|\eta| < 0.5)$"
  )
  plt.ylabel(
      r"$\langle N_{D} \rangle_{\text{local}} / \langle N_{D}"
      r" \rangle_{\text{global}}$"
  )
  plt.grid(True, linestyle="--", alpha=0.7)
  plt.legend(
      loc="upper left",
      frameon=False,
      title=r"All Physics ($X$ fixo: $|\eta| < 0.5$)",
      title_fontsize=11,
      fontsize=10,
  )
  plt.tight_layout()

  caminho_fig_x_fixo = os.path.join(
      base_figuras_dir, "Correlacao_QQ_todos_eta_x_fixed_0p5.pdf"
  )
  plt.savefig(caminho_fig_x_fixo, bbox_inches="tight")
  plt.close()

  log_print(
      f"[OK] Gráfico com X fixo em |\eta| < 0.5 salvo em: {caminho_fig_x_fixo}"
  )

log_print(
    f"\n{'='*120}\nSucesso! Todos os plots foram salvos em:"
    f" {base_figuras_dir}\n{'='*120}"
)
log_file.close()