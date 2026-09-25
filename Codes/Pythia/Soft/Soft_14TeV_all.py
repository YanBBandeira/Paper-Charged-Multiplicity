# =================================================================
# SCRIPT DE SIMULAÇÃO PYTHIA 8 - FÍSICA DE PRÓTON-PRÓTON (pp)
# Simulação de colisões pp a 14 TeV com análise cinemática diferencial e global
# =================================================================

import os
import pythia8
import numpy as np

# =================================================================
# 1. CONFIGURAÇÕES INICIAIS E DIRETÓRIOS DE SAÍDA
# =================================================================

# Tenta extrair o nome do script atual automaticamente para usar nos arquivos
try:
    nome_script = os.path.splitext(os.path.basename(__file__))[0]
except NameError:
    nome_script = "simulacao_pp"

# Pega a última parte do nome do script para definir a subpasta de destino (ex: 'Hard')
subpasta_nome = nome_script.split("_")[-1]

# -----------------------------------------------------------------
# Definição dos Cortes Cinemáticos (Pseudorapidez e Momento Transverso)
# -----------------------------------------------------------------
# Janelas de corte para a pseudorapidez (|eta| < corte)
cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

# Faixas diferenciais de momento transverso (pT_min, pT_max, identificador_para_arquivo)
faixas_pt = [
    (1.0,    2.0, "pt_1_2"),     # 1 < pT < 2 GeV/c
    (2.0,    4.0, "pt_2_4"),     # 2 < pT < 4 GeV/c
    (4.0,    6.0, "pt_4_6"),     # 4 < pT < 6 GeV/c
    (6.0,    8.0, "pt_6_8"),     # 6 < pT < 8 GeV/c
    (8.0,   12.0, "pt_8_12"),    # 8 < pT < 12 GeV/c
    (12.0,  24.0, "pt_12_24")    # 12 < pT < 24 GeV/c
]

# Códigos PDG dos hástrons pesados e partículas carregadas de interesse
# Nota: Utilizamos a última cópia física das ressonâncias para evitar contagem dupla
pdg_D_mesons  = [411]   # Mésons D+ e D-
pdg_Ds_mesons = [431]   # Mésons Ds+ e Ds-
pdg_D0_mesons = [421]   # Mésons D0 e D0bar

# Parâmetros gerais dos histogramas e do número de eventos
bin_width = 0.1
nEvents = 1000000

# Criação estruturada do diretório de saída para os arquivos de dados (.dat)
output_dir = os.path.join("..", "..", "Dados", "Soft", subpasta_nome)
os.makedirs(output_dir, exist_ok=True)

# =================================================================
# 2. INICIALIZAÇÃO E CONFIGURAÇÃO DO GERADOR PYTHIA 8
# =================================================================
pythia = pythia8.Pythia()
pythia.readString("Random:setSeed = on")           # Ativa semente aleatória controlada
pythia.readString("Random:seed = 42")              # Semente fixa para reprodutibilidade
pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:idB = 2212")
pythia.readString("Beams:eCM = 14.e3")             # Energia no centro de massa: 14 TeV
pythia.readString("SoftQCD:all = on")             # Desliga processos suaves (soft QCD)
pythia.readString("HardQCD:all = off")              # Liga processos duros (hard scattering)
pythia.readString("PartonLevel:ISR = on")          # Radiação no estado inicial (ISR)
pythia.readString("PartonLevel:MPI = on")          # Interações Múltiplas de Partons (MPI)
pythia.readString("PartonLevel:FSR = on")          # Radiação no estado final (FSR)
pythia.init()                                      # Inicializa a engine do Pythia

# =================================================================
# 3. DECLARAÇÃO DE HISTOGRAMAS E ESTRUTURAS DE DADOS
# =================================================================

# Listas principais globais integradas indexadas pelo corte de eta
h_eta_ch, h_eta_d0, h_eta_d, h_eta_ds = [], [], [], []
h_pt_ch,  h_pt_d0,  h_pt_d,  h_pt_ds  = [], [], [], []

# Histogramas de Multiplicidade Global Integrada (pT > 0) por corte de eta
h_mult_ch_global = []
h_mult_d0_global = []
h_mult_d_global  = []
h_mult_ds_global = []

# Matrizes 2D para a densidade diferencial bidimensional d2N / (deta dpt) global
matrizes_d2n_ch = []
matrizes_d2n_d0 = []
matrizes_d2n_d  = []
matrizes_d2n_ds = []

# Listas auxiliares para armazenar os limites dos bins numéricos
bins_eta_lista = []
bins_pt_lista  = []

# Listas estruturadas para guardar os histogramas diferenciais por faixa de pT
h_eta_ch_faixas_pt = []
h_eta_d0_faixas_pt = []
h_eta_d_faixas_pt  = []
h_eta_ds_faixas_pt = []

h_mult_ch_faixas_pt = []
h_mult_d0_faixas_pt = []
h_mult_d_faixas_pt  = []
h_mult_ds_faixas_pt = []

# Estruturas para armazenar os eventos brutos (N_ch, N_D0, N_D, N_Ds)
eventos_por_eta_global = []          # Casos globais (pT > 0) por janela de eta
eventos_por_eta_faixas_pt = []       # Casos diferenciais por janela de eta e faixa de pT

# Loop de configuração para instanciar os histogramas e estruturas de forma procedural
for i_corte, corte in enumerate(cortes_eta):
    n_bins_eta = int(round(2.0 * corte / bin_width))
    pt_max_hist = 150.0  
    n_bins_pt  = int(round(pt_max_hist / bin_width))  
    
    # Criação dos vetores de limites (bins) com NumPy
    bins_eta = np.linspace(-corte, corte, n_bins_eta + 1)
    bins_pt = np.linspace(0.0, pt_max_hist, n_bins_pt + 1)
    
    bins_eta_lista.append(bins_eta)
    bins_pt_lista.append(bins_pt)
    
    # Inicializa matrizes 2D globais preenchidas com zeros para a distribuição (eta vs pT)
    matrizes_d2n_ch.append(np.zeros((n_bins_eta, n_bins_pt)))
    matrizes_d2n_d0.append(np.zeros((n_bins_eta, n_bins_pt)))
    matrizes_d2n_d.append(np.zeros((n_bins_eta, n_bins_pt)))
    matrizes_d2n_ds.append(np.zeros((n_bins_eta, n_bins_pt)))
    
    # Histogramas globais integrados por janela de eta
    h_eta_ch.append(pythia8.Hist(f"dN/deta Carregadas |eta|<{corte}", n_bins_eta, -corte, corte))
    h_eta_d0.append(pythia8.Hist(f"dN/deta D0 |eta|<{corte}", n_bins_eta, -corte, corte))
    h_eta_d.append(pythia8.Hist(f"dN/deta D+ |eta|<{corte}", n_bins_eta, -corte, corte))
    h_eta_ds.append(pythia8.Hist(f"dN/deta Ds |eta|<{corte}", n_bins_eta, -corte, corte))
    
    h_pt_ch.append(pythia8.Hist(f"pT Carregadas |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))
    h_pt_d0.append(pythia8.Hist(f"pT D0 |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))
    h_pt_d.append(pythia8.Hist(f"pT D+ |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))
    h_pt_ds.append(pythia8.Hist(f"pT Ds |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))

    # Histogramas de Multiplicidade Global Integrada por janela de eta
    h_mult_ch_global.append(pythia8.Hist(f"Mult Global Ch |eta|<{corte}", 500, -0.5, 499.5))
    h_mult_d0_global.append(pythia8.Hist(f"Mult Global D0 |eta|<{corte}", 200, -0.5, 199.5))
    h_mult_d_global.append(pythia8.Hist(f"Mult Global D+ |eta|<{corte}", 50, -0.5, 49.5))
    h_mult_ds_global.append(pythia8.Hist(f"Mult Global Ds |eta|<{corte}", 50, -0.5, 49.5))

    # Inicializa a lista de eventos brutos globais para este corte de eta
    eventos_por_eta_global.append([])

    # Sub-listas temporárias para armazenar os elementos específicos de cada faixa de pT diferencial
    sub_eta_ch, sub_eta_d0, sub_eta_d, sub_eta_ds = [], [], [], []
    sub_mult_ch, sub_mult_d0, sub_mult_d, sub_mult_ds = [], [], [], []
    sub_eventos_faixas = []

    for pt_min, pt_max, pt_label in faixas_pt:
        # Histogramas dN/deta restritos à faixa de pT correspondente
        sub_eta_ch.append(pythia8.Hist(f"dN/deta Ch |eta|<{corte} {pt_label}", n_bins_eta, -corte, corte))
        sub_eta_d0.append(pythia8.Hist(f"dN/deta D0 |eta|<{corte} {pt_label}", n_bins_eta, -corte, corte))
        sub_eta_d.append(pythia8.Hist(f"dN/deta D+ |eta|<{corte} {pt_label}", n_bins_eta, -corte, corte))
        sub_eta_ds.append(pythia8.Hist(f"dN/deta Ds |eta|<{corte} {pt_label}", n_bins_eta, -corte, corte))

        # Histogramas de Multiplicidade restritos à faixa de pT correspondente
        sub_mult_ch.append(pythia8.Hist(f"Mult Ch |eta|<{corte} {pt_label}", 500, -0.5, 499.5))
        sub_mult_d0.append(pythia8.Hist(f"Mult D0 |eta|<{corte} {pt_label}", 200, -0.5, 199.5))
        sub_mult_d.append(pythia8.Hist(f"Mult D+ |eta|<{corte} {pt_label}", 50, -0.5, 49.5))
        sub_mult_ds.append(pythia8.Hist(f"Mult Ds |eta|<{corte} {pt_label}", 50, -0.5, 49.5))

        # Inicializa a lista vazia de eventos brutos para esta faixa específica de pT
        sub_eventos_faixas.append([])

    # Associa as sub-listas às listas principais por corte de eta
    h_eta_ch_faixas_pt.append(sub_eta_ch)
    h_eta_d0_faixas_pt.append(sub_eta_d0)
    h_eta_d_faixas_pt.append(sub_eta_d)
    h_eta_ds_faixas_pt.append(sub_eta_ds)

    h_mult_ch_faixas_pt.append(sub_mult_ch)
    h_mult_d0_faixas_pt.append(sub_mult_d0)
    h_mult_d_faixas_pt.append(sub_mult_d)
    h_mult_ds_faixas_pt.append(sub_mult_ds)

    eventos_por_eta_faixas_pt.append(sub_eventos_faixas)

# Contador de eventos aceitos com sucesso pela simulação
n_accepted = 0

# =================================================================
# 4. LOOP PRINCIPAL DE PROCESSAMENTO DE EVENTOS
# =================================================================
for iEvt in range(nEvents):
    # Gera o próximo evento físico no Pythia. Se falhar, avança para o próximo ciclo.
    if not pythia.next():
        continue

    n_accepted += 1

    # =================================================================
    # Inicialização de contadores temporários por evento (Global e Diferencial)
    # =================================================================
    cont_ch_global = [0 for _ in cortes_eta]
    cont_d0_global = [0 for _ in cortes_eta]
    cont_d_global  = [0 for _ in cortes_eta]
    cont_ds_global = [0 for _ in cortes_eta]

    cont_ch_evento = [[0 for _ in faixas_pt] for _ in cortes_eta]
    cont_d0_evento = [[0 for _ in faixas_pt] for _ in cortes_eta]
    cont_d_evento  = [[0 for _ in faixas_pt] for _ in cortes_eta]
    cont_ds_evento = [[0 for _ in faixas_pt] for _ in cortes_eta]

    # Itera por todas as partículas geradas no evento atual do Pythia
    for i in range(pythia.event.size()):
        p = pythia.event[i]
        
        # -----------------------------------------------------------------
        # Algoritmo de seleção da última cópia (Last Copy) de ressonâncias
        # Garante que pegamos o estado final da partícula antes do decaimento
        # -----------------------------------------------------------------
        d1, d2 = p.daughter1(), p.daughter2()
        tem_filha_igual = False
        if d1 > 0:
            for d in range(d1, d2 + 1):
                if pythia.event[d].id() == p.id():
                    tem_filha_igual = True
                    break
        last_copy = not tem_filha_igual 
        
        # Extração das grandezas cinemáticas fundamentais da partícula
        pdg_abs = abs(p.id())
        eta = p.eta()
        eta_abs = abs(eta)
        pt = p.pT()
        
        # Identificação booleana das espécies de partículas exigidas
        is_charged  = (p.isCharged() and p.isFinal())
        is_meson_D0 = (pdg_abs in pdg_D0_mesons and last_copy)
        is_meson_D  = (pdg_abs in pdg_D_mesons and last_copy)
        is_meson_Ds = (pdg_abs in pdg_Ds_mesons and last_copy)
        
        # Se a partícula não pertencer a nenhum grupo de interesse, ignora
        if not (is_charged or is_meson_D0 or is_meson_D or is_meson_Ds):
            continue

        # -----------------------------------------------------------------
        # Varredura por janelas de pseudorapidez e faixas de momento transverso
        # -----------------------------------------------------------------
        for i_corte, corte in enumerate(cortes_eta):
            # Verifica se a partícula está dentro do limite geométrico de eta atual
            if eta_abs < corte:
                
                # Preenchimento dos histogramas globais integrados padrão
                if is_charged:  
                    h_eta_ch[i_corte].fill(eta)
                    h_pt_ch[i_corte].fill(pt)
                    cont_ch_global[i_corte] += 1
                if is_meson_D0: 
                    h_eta_d0[i_corte].fill(eta)
                    h_pt_d0[i_corte].fill(pt)
                    cont_d0_global[i_corte] += 1
                if is_meson_D:  
                    h_eta_d[i_corte].fill(eta)
                    h_pt_d[i_corte].fill(pt)
                    cont_d_global[i_corte] += 1
                if is_meson_Ds:  
                    h_eta_ds[i_corte].fill(eta)
                    h_pt_ds[i_corte].fill(pt)
                    cont_ds_global[i_corte] += 1

                # Avaliação de qual faixa diferencial de pT a partícula pertence
                for i_faixa, (pt_min, pt_max, _) in enumerate(faixas_pt):
                    if pt_min < pt < pt_max:
                        # Preenche dN/deta diferencial por faixa de pT e incrementa contador do evento
                        if is_charged:  
                            h_eta_ch_faixas_pt[i_corte][i_faixa].fill(eta)
                            cont_ch_evento[i_corte][i_faixa] += 1
                        if is_meson_D0: 
                            h_eta_d0_faixas_pt[i_corte][i_faixa].fill(eta)
                            cont_d0_evento[i_corte][i_faixa] += 1
                        if is_meson_D:  
                            h_eta_d_faixas_pt[i_corte][i_faixa].fill(eta)
                            cont_d_evento[i_corte][i_faixa] += 1
                        if is_meson_Ds: 
                            h_eta_ds_faixas_pt[i_corte][i_faixa].fill(eta)
                            cont_ds_evento[i_corte][i_faixa] += 1

                # Acumulação rápida e otimizada da Matriz 2D global d2N / (deta dpt)
                bins_e = bins_eta_lista[i_corte]
                bins_p = bins_pt_lista[i_corte]
                
                if bins_e[0] <= eta <= bins_e[-1] and 0.0 <= pt <= bins_p[-1]:
                    b_eta = int((eta - bins_e[0]) / bin_width)
                    b_pt  = int(pt / bin_width)
                    
                    if 0 <= b_eta < len(bins_e) - 1 and 0 <= b_pt < len(bins_p) - 1:
                        if is_charged:  
                            matrizes_d2n_ch[i_corte][b_eta, b_pt] += 1.0
                        if is_meson_D0: 
                            matrizes_d2n_d0[i_corte][b_eta, b_pt] += 1.0
                        if is_meson_D:  
                            matrizes_d2n_d[i_corte][b_eta, b_pt]  += 1.0
                        if is_meson_Ds: 
                            matrizes_d2n_ds[i_corte][b_eta, b_pt] += 1.0

    # -----------------------------------------------------------------
    # Preenchimento dos histogramas de multiplicidade e registro de eventos brutos (Global e Diferencial)
    # -----------------------------------------------------------------
    for i_corte in range(len(cortes_eta)):
        # Preenchimento global (pT > 0)
        h_mult_ch_global[i_corte].fill(cont_ch_global[i_corte])
        h_mult_d0_global[i_corte].fill(cont_d0_global[i_corte])
        h_mult_d_global[i_corte].fill(cont_d_global[i_corte])
        h_mult_ds_global[i_corte].fill(cont_ds_global[i_corte])

        eventos_por_eta_global[i_corte].append([
            cont_ch_global[i_corte],
            cont_d0_global[i_corte],
            cont_d_global[i_corte],
            cont_ds_global[i_corte]
        ])

        # Preenchimento por faixas diferenciais de pT
        for i_faixa in range(len(faixas_pt)):
            h_mult_ch_faixas_pt[i_corte][i_faixa].fill(cont_ch_evento[i_corte][i_faixa])
            h_mult_d0_faixas_pt[i_corte][i_faixa].fill(cont_d0_evento[i_corte][i_faixa])
            h_mult_d_faixas_pt[i_corte][i_faixa].fill(cont_d_evento[i_corte][i_faixa])
            h_mult_ds_faixas_pt[i_corte][i_faixa].fill(cont_ds_evento[i_corte][i_faixa])

            eventos_por_eta_faixas_pt[i_corte][i_faixa].append([
                cont_ch_evento[i_corte][i_faixa],
                cont_d0_evento[i_corte][i_faixa],
                cont_d_evento[i_corte][i_faixa],
                cont_ds_evento[i_corte][i_faixa]
            ])

    # Exibe no console uma mensagem de progresso a cada 2000 eventos processados
    if n_accepted % 2000 == 0:
        print(f"Progresso: {n_accepted} eventos processados com sucesso.")

# Imprime o sumário de estatísticas gerais gerado pela biblioteca Pythia
pythia.stat()

# =================================================================
# 5. ESCRITA E EXPORTAÇÃO DOS DADOS EM ARQUIVOS DE TEXTO (.dat)
# =================================================================

# 1. Exporta histogramas diferenciais, de multiplicidade e tabelas de eventos brutos por faixa de pT e corte de eta
for i_corte, corte in enumerate(cortes_eta):
    corte_str = f"{corte:.1f}".replace(".", "p")
    
    for i_faixa, (_, _, pt_label) in enumerate(faixas_pt):
        # Nomes dos arquivos de dN/deta diferencial por pT
        h_eta_ch_faixas_pt[i_corte][i_faixa].table(os.path.join(output_dir, f"{nome_script}_dndeta_ch_eta_{corte_str}_{pt_label}.dat"))
        h_eta_d0_faixas_pt[i_corte][i_faixa].table(os.path.join(output_dir, f"{nome_script}_dndeta_d0_eta_{corte_str}_{pt_label}.dat"))
        h_eta_d_faixas_pt[i_corte][i_faixa].table(os.path.join(output_dir, f"{nome_script}_dndeta_d_eta_{corte_str}_{pt_label}.dat"))
        h_eta_ds_faixas_pt[i_corte][i_faixa].table(os.path.join(output_dir, f"{nome_script}_dndeta_ds_eta_{corte_str}_{pt_label}.dat"))

        # Nomes dos arquivos de multiplicidade diferencial por pT
        h_mult_ch_faixas_pt[i_corte][i_faixa].table(os.path.join(output_dir, f"{nome_script}_mult_ch_eta_{corte_str}_{pt_label}.dat"))
        h_mult_d0_faixas_pt[i_corte][i_faixa].table(os.path.join(output_dir, f"{nome_script}_mult_d0_eta_{corte_str}_{pt_label}.dat"))
        h_mult_d_faixas_pt[i_corte][i_faixa].table(os.path.join(output_dir, f"{nome_script}_mult_d_eta_{corte_str}_{pt_label}.dat"))
        h_mult_ds_faixas_pt[i_corte][i_faixa].table(os.path.join(output_dir, f"{nome_script}_mult_ds_eta_{corte_str}_{pt_label}.dat"))

        # Salvamento do arquivo de eventos brutos diferenciais (N_ch, N_D0, N_D, N_Ds)
        c_events_pt = os.path.join(output_dir, f"{nome_script}_events_eta_{corte_str}_{pt_label}.dat")
        np.savetxt(
            c_events_pt,
            eventos_por_eta_faixas_pt[i_corte][i_faixa],
            fmt='%d %d %d %d',
            header="N_ch N_D0 N_D N_Ds",
            comments=''
        )

# 2. Exporta os histogramas globais integrados, matrizes 2D, multiplicidades globais e eventos globais por corte de eta
for i_corte, corte in enumerate(cortes_eta):
    corte_str = f"{corte:.1f}".replace(".", "p")

    # Escrita das distribuições globais integradas dN/deta
    h_eta_ch[i_corte].table(os.path.join(output_dir, f"{nome_script}_dndeta_ch_eta_{corte_str}.dat"))
    h_eta_d0[i_corte].table(os.path.join(output_dir, f"{nome_script}_dndeta_d0_eta_{corte_str}.dat"))
    h_eta_d[i_corte].table(os.path.join(output_dir, f"{nome_script}_dndeta_d_eta_{corte_str}.dat"))
    h_eta_ds[i_corte].table(os.path.join(output_dir, f"{nome_script}_dndeta_ds_eta_{corte_str}.dat"))
    
    # Escrita das distribuições globais integradas de momento transverso (dN/dpt)
    h_pt_ch[i_corte].table(os.path.join(output_dir, f"{nome_script}_pt_ch_eta_{corte_str}.dat"))
    h_pt_d0[i_corte].table(os.path.join(output_dir, f"{nome_script}_pt_d0_eta_{corte_str}.dat"))
    h_pt_d[i_corte].table(os.path.join(output_dir, f"{nome_script}_pt_d_eta_{corte_str}.dat"))
    h_pt_ds[i_corte].table(os.path.join(output_dir, f"{nome_script}_pt_ds_eta_{corte_str}.dat"))

    # Escrita dos histogramas de Multiplicidade Global Integrada (pT > 0)
    h_mult_ch_global[i_corte].table(os.path.join(output_dir, f"{nome_script}_mult_ch_eta_{corte_str}_global.dat"))
    h_mult_d0_global[i_corte].table(os.path.join(output_dir, f"{nome_script}_mult_d0_eta_{corte_str}_global.dat"))
    h_mult_d_global[i_corte].table(os.path.join(output_dir, f"{nome_script}_mult_d_eta_{corte_str}_global.dat"))
    h_mult_ds_global[i_corte].table(os.path.join(output_dir, f"{nome_script}_mult_ds_eta_{corte_str}_global.dat"))

    # Escrita do arquivo de eventos brutos globais (N_ch, N_D0, N_D, N_Ds) por janela de eta
    c_events_global = os.path.join(output_dir, f"{nome_script}_events_eta_{corte_str}_global.dat")
    np.savetxt(
        c_events_global,
        eventos_por_eta_global[i_corte],
        fmt='%d %d %d %d',
        header="N_ch N_D0 N_D N_Ds",
        comments=''
    )

    # Função interna procedural para normalizar e exportar a matriz 2D global d2N / (deta dpt)
    def salvar_matriz_2d(matriz, nome_base_arq):
        bins_e = bins_eta_lista[i_corte]
        bins_p = bins_pt_lista[i_corte]
        d_eta = bins_e[1] - bins_e[0]
        d_pt  = bins_p[1] - bins_p[0]
        
        dados_tabulados = []
        for ie in range(matriz.shape[0]):
            val_eta = 0.5 * (bins_e[ie] + bins_e[ie+1])
            for ip in range(matriz.shape[1]):
                val_pt = 0.5 * (bins_p[ip] + bins_p[ip+1])
                # Normalização pela quantidade total de eventos aceitos e larguras dos bins
                val_d2n = matriz[ie, ip] / (n_accepted * d_eta * d_pt) if n_accepted > 0 else 0.0
                dados_tabulados.append([val_eta, val_pt, val_d2n])
                
        np.savetxt(
            os.path.join(output_dir, nome_base_arq),
            dados_tabulados,
            fmt="%.6e %.6e %.6e",
            header="Eta  pT  d2N_deta_dpt",
            comments=""
        )

    # Executa a gravação das matrizes 2D globais para cada espécie particulada
    salvar_matriz_2d(matrizes_d2n_ch[i_corte], f"{nome_script}_d2n_etapt_ch_eta_{corte_str}.dat")
    salvar_matriz_2d(matrizes_d2n_d0[i_corte], f"{nome_script}_d2n_etapt_d0_eta_{corte_str}.dat")
    salvar_matriz_2d(matrizes_d2n_d[i_corte],  f"{nome_script}_d2n_etapt_d_eta_{corte_str}.dat")
    salvar_matriz_2d(matrizes_d2n_ds[i_corte], f"{nome_script}_d2n_etapt_ds_eta_{corte_str}.dat")

print(f"\nSimulação concluída com sucesso!")
print(f"Total de eventos válidos computados: {n_accepted}")
print(f"Todos os arquivos globais e diferenciais foram salvos em: {output_dir}")