import os
import pythia8
import numpy as np

# =================================================================
# 1. Configurações Iniciais e Nomes de Arquivos
# =================================================================
try:
    nome_script = os.path.splitext(os.path.basename(__file__))[0]
except NameError:
    nome_script = "simulacao_pp"

# Extrai o texto após o último "_" para criar a subpasta
subpasta_nome = nome_script.split("_")[-1]

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
cortes_pt_dndeta = [5.0, 10.0, 20.0, 40.0]  # Cortes superiores de pT: pT < pt_corte

pdg_D_mesons = [411]            # D+
pdg_Ds_mesons = [431]           # Ds+
pdg_K_mesons = [321]            # Kaons
bin_width = 0.1
nEvents = 1000000

# Caminho com a subpasta dinâmica baseada no título do script
output_dir = os.path.join("..", "..", "Dados", "Hard", subpasta_nome)
os.makedirs(output_dir, exist_ok=True)

# =================================================================
# 2. Inicializar o Pythia
# =================================================================
pythia = pythia8.Pythia()
pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:eCM = 14.e3")
pythia.readString("SoftQCD:all = off")
pythia.readString("HardQCD:all = on")
pythia.readString("PartonLevel:ISR = off") 
pythia.readString("PartonLevel:MPI = off")
pythia.readString("PartonLevel:FSR = on") 
pythia.init()

# =================================================================
# 3. Histogramas Nativos e Estruturas de Armazenamento
# =================================================================
h_mult_ch, h_mult_k, h_mult_d, h_mult_ds = [], [], [], []
h_eta_ch,  h_eta_k,  h_eta_d,  h_eta_ds  = [], [], [], []
h_pt_ch,   h_pt_k,   h_pt_d,   h_pt_ds   = [], [], [], []

# Matriz para guardar (N_ch, N_D, N_Ds, N_K) de cada evento aceito por janela de eta
eventos_por_eta = [[] for _ in cortes_eta]

# Histogramas dN/deta com corte superior em pT (pT < pt_corte)
h_eta_d_pt  = []
h_eta_ds_pt = []
h_eta_ch_pt = []
h_eta_k_pt  = []

# Estruturas 2D em NumPy para d2N / (deta dpt) para cada corte de eta
matrizes_d2n_ch = []
matrizes_d2n_k  = []
matrizes_d2n_d  = []
matrizes_d2n_ds = []

bins_eta_lista = []
bins_pt_lista  = []

for pt_corte in cortes_pt_dndeta:
    h_eta_d_pt.append(pythia8.Hist(f"dN/deta Mesons D pt<{pt_corte}GeV", 100, -5.0, 5.0))
    h_eta_ds_pt.append(pythia8.Hist(f"dN/deta Mesons Ds pt<{pt_corte}GeV", 100, -5.0, 5.0))
    h_eta_ch_pt.append(pythia8.Hist(f"dN/deta Carregadas pt<{pt_corte}GeV", 100, -5.0, 5.0))
    h_eta_k_pt.append(pythia8.Hist(f"dN/deta Kaons pt<{pt_corte}GeV", 100, -5.0, 5.0))

# Histogramas 1D para cada janela de eta
for corte in cortes_eta:
    n_bins_eta = int(round(2.0 * corte / bin_width))
    pt_max_hist = 150.0  
    n_bins_pt  = int(round(pt_max_hist / bin_width))  
    
    bins_eta = np.linspace(-corte, corte, n_bins_eta + 1)
    bins_pt = np.linspace(0.0, pt_max_hist, n_bins_pt + 1)
    
    bins_eta_lista.append(bins_eta)
    bins_pt_lista.append(bins_pt)
    
    matrizes_d2n_ch.append(np.zeros((n_bins_eta, n_bins_pt)))
    matrizes_d2n_k.append(np.zeros((n_bins_eta, n_bins_pt)))
    matrizes_d2n_d.append(np.zeros((n_bins_eta, n_bins_pt)))
    matrizes_d2n_ds.append(np.zeros((n_bins_eta, n_bins_pt)))
    
    # dN/deta
    h_eta_ch.append(pythia8.Hist(f"dN/deta Carregadas |eta|<{corte}", n_bins_eta, -corte, corte))
    h_eta_k.append(pythia8.Hist(f"dN/deta Kaons |eta|<{corte}", n_bins_eta, -corte, corte))
    h_eta_d.append(pythia8.Hist(f"dN/deta Mesons D |eta|<{corte}", n_bins_eta, -corte, corte))
    h_eta_ds.append(pythia8.Hist(f"dN/deta Mesons Ds |eta|<{corte}", n_bins_eta, -corte, corte))
    
    # dN/dpt
    h_pt_ch.append(pythia8.Hist(f"pT Carregadas |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))
    h_pt_k.append(pythia8.Hist(f"pT Kaons |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))
    h_pt_d.append(pythia8.Hist(f"pT Mesons D |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))
    h_pt_ds.append(pythia8.Hist(f"pT Mesons Ds |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))

    # Multiplicidade
    h_mult_ch.append(pythia8.Hist(f"Mult Carregadas |eta|<{corte}", 400, -0.5, 399.5)) 
    h_mult_k.append(pythia8.Hist(f"Mult Kaons |eta|<{corte}", 100, -0.5, 99.5))
    h_mult_d.append(pythia8.Hist(f"Mult Mesons D |eta|<{corte}", 50, -0.5, 49.5))
    h_mult_ds.append(pythia8.Hist(f"Mult Mesons Ds |eta|<{corte}", 50, -0.5, 49.5))

n_accepted = 0

# =================================================================
# 4. Loop Principal de Eventos
# =================================================================
for iEvt in range(nEvents):
    if not pythia.next():
        continue

    n_accepted += 1

    # Inicializa os contadores para as partículas em cada janela de eta
    n_ch_evento = [0] * len(cortes_eta)
    n_K_evento  = [0] * len(cortes_eta)
    n_D_evento  = [0] * len(cortes_eta)
    n_Ds_evento = [0] * len(cortes_eta)

    # Varre as partículas do evento aceito
    for i in range(pythia.event.size()):
        p = pythia.event[i]
        
        # Garante a escolha da última cópia de ressonâncias
        d1, d2 = p.daughter1(), p.daughter2()
        tem_filha_igual = False
        if d1 > 0:
            for d in range(d1, d2 + 1):
                if pythia.event[d].id() == p.id():
                    tem_filha_igual = True
                    break
        last_copy = not tem_filha_igual 
        
        pdg_abs = abs(p.id())
        eta = p.eta()
        eta_abs = abs(eta)
        pt = p.pT()
        
        is_charged  = (p.isCharged() and p.isFinal())
        is_kaon     = (pdg_abs in pdg_K_mesons and last_copy)
        is_meson_D  = (pdg_abs in pdg_D_mesons and last_copy)
        is_meson_Ds = (pdg_abs in pdg_Ds_mesons and last_copy)
        
        # Se a partícula não for de interesse, pula para a próxima
        if not (is_charged or is_kaon or is_meson_D or is_meson_Ds):
            continue

        # dN/deta com corte pt < pt_corte para todas as seleções
        if is_meson_D:
            for i_pt, pt_corte in enumerate(cortes_pt_dndeta):
                if pt < pt_corte:
                    h_eta_d_pt[i_pt].fill(eta)

        if is_meson_Ds:
            for i_pt, pt_corte in enumerate(cortes_pt_dndeta):
                if pt < pt_corte:
                    h_eta_ds_pt[i_pt].fill(eta)
                            
        if is_charged:
            for i_pt, pt_corte in enumerate(cortes_pt_dndeta):
                if pt < pt_corte:
                    h_eta_ch_pt[i_pt].fill(eta)
                            
        if is_kaon:
            for i_pt, pt_corte in enumerate(cortes_pt_dndeta):
                if pt < pt_corte:
                    h_eta_k_pt[i_pt].fill(eta)

        # Preenche os histogramas 1D, pT, multiplicidades e matrizes 2D por corte de eta
        for i_corte, corte in enumerate(cortes_eta):
            if eta_abs < corte:
                if is_charged:
                    h_eta_ch[i_corte].fill(eta)
                    h_pt_ch[i_corte].fill(pt)
                    n_ch_evento[i_corte] += 1
                
                if is_kaon:
                    h_eta_k[i_corte].fill(eta)
                    h_pt_k[i_corte].fill(pt)
                    n_K_evento[i_corte] += 1
                
                if is_meson_D:
                    h_eta_d[i_corte].fill(eta)
                    h_pt_d[i_corte].fill(pt)
                    n_D_evento[i_corte] += 1
                
                if is_meson_Ds:
                    h_eta_ds[i_corte].fill(eta)
                    h_pt_ds[i_corte].fill(pt)
                    n_Ds_evento[i_corte] += 1
                    
            # Acumulação da Matriz 2D para d2N / (deta dpt) baseada nos bins específicos do corte
            b_eta = np.digitize(eta, bins_eta_lista[i_corte]) - 1
            b_pt  = np.digitize(pt, bins_pt_lista[i_corte]) - 1
            
            n_b_eta = len(bins_eta_lista[i_corte]) - 1
            n_b_pt  = len(bins_pt_lista[i_corte]) - 1
            
            if 0 <= b_eta < n_b_eta and 0 <= b_pt < n_b_pt:
                if is_charged:  matrizes_d2n_ch[i_corte][b_eta, b_pt] += 1.0
                if is_kaon:     matrizes_d2n_k[i_corte][b_eta, b_pt]  += 1.0
                if is_meson_D:  matrizes_d2n_d[i_corte][b_eta, b_pt]  += 1.0
                if is_meson_Ds: matrizes_d2n_ds[i_corte][b_eta, b_pt] += 1.0

    # Atualiza histogramas de multiplicidade e salva a tupla do evento
    for i_corte in range(len(cortes_eta)):
        h_mult_ch[i_corte].fill(n_ch_evento[i_corte])
        h_mult_k[i_corte].fill(n_K_evento[i_corte])
        h_mult_d[i_corte].fill(n_D_evento[i_corte])
        h_mult_ds[i_corte].fill(n_Ds_evento[i_corte])

        eventos_por_eta[i_corte].append([n_ch_evento[i_corte], n_D_evento[i_corte], n_Ds_evento[i_corte], n_K_evento[i_corte]])

    if n_accepted % 2000 == 0:
        print(f"Eventos processados: {n_accepted}")

pythia.stat()

# =================================================================
# 5. Escrita dos Dados em Arquivos de Texto (.dat)
# =================================================================

# Tabelas de dN/deta com cortes superiores em pT (pt < pt_corte)
for i_pt, pt_corte in enumerate(cortes_pt_dndeta):
    pt_str = f"{pt_corte:.0f}"
    h_eta_d_pt[i_pt].table(os.path.join(output_dir, f"{nome_script}_dndeta_d_pt_lt_{pt_str}GeV.dat"))
    h_eta_ds_pt[i_pt].table(os.path.join(output_dir, f"{nome_script}_dndeta_ds_pt_lt_{pt_str}GeV.dat"))
    h_eta_ch_pt[i_pt].table(os.path.join(output_dir, f"{nome_script}_dndeta_ch_pt_lt_{pt_str}GeV.dat"))
    h_eta_k_pt[i_pt].table(os.path.join(output_dir, f"{nome_script}_dndeta_k_pt_lt_{pt_str}GeV.dat"))

# Tabelas 1D, 2D e dados brutos por corte de eta
for i_corte, corte in enumerate(cortes_eta):
    corte_str = f"{corte:.1f}".replace(".", "p")

    # Multiplicidade
    h_mult_ch[i_corte].table(os.path.join(output_dir, f"{nome_script}_mult_ch_eta_{corte_str}.dat"))
    h_mult_k[i_corte].table(os.path.join(output_dir, f"{nome_script}_mult_k_eta_{corte_str}.dat"))
    h_mult_d[i_corte].table(os.path.join(output_dir, f"{nome_script}_mult_d_eta_{corte_str}.dat"))
    h_mult_ds[i_corte].table(os.path.join(output_dir, f"{nome_script}_mult_ds_eta_{corte_str}.dat"))

    # dN/deta
    h_eta_ch[i_corte].table(os.path.join(output_dir, f"{nome_script}_dndeta_ch_eta_{corte_str}.dat"))
    h_eta_k[i_corte].table(os.path.join(output_dir, f"{nome_script}_dndeta_k_eta_{corte_str}.dat"))
    h_eta_d[i_corte].table(os.path.join(output_dir, f"{nome_script}_dndeta_d_eta_{corte_str}.dat"))
    h_eta_ds[i_corte].table(os.path.join(output_dir, f"{nome_script}_dndeta_ds_eta_{corte_str}.dat"))
    
    # pT
    h_pt_ch[i_corte].table(os.path.join(output_dir, f"{nome_script}_pt_ch_eta_{corte_str}.dat"))
    h_pt_k[i_corte].table(os.path.join(output_dir, f"{nome_script}_pt_k_eta_{corte_str}.dat"))
    h_pt_d[i_corte].table(os.path.join(output_dir, f"{nome_script}_pt_d_eta_{corte_str}.dat"))
    h_pt_ds[i_corte].table(os.path.join(output_dir, f"{nome_script}_pt_ds_eta_{corte_str}.dat"))

    # Função auxiliar para salvar matriz 2D no formato tabular (eta, pt, d2N/deta dpt)
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
                # Normalização diferencial dividindo por n_accepted e larguras de bin
                val_d2n = matriz[ie, ip] / (n_accepted * d_eta * d_pt) if n_accepted > 0 else 0.0
                dados_tabulados.append([val_eta, val_pt, val_d2n])
                
        np.savetxt(
            os.path.join(output_dir, nome_base_arq),
            dados_tabulados,
            fmt="%.6e %.6e %.6e",
            header="Eta  pT  d2N_deta_dpt",
            comments=""
        )

    # Salvando as matrizes 2D d2N / (deta dpt)
    salvar_matriz_2d(matrizes_d2n_ch[i_corte], f"{nome_script}_d2n_etapt_ch_eta_{corte_str}.dat")
    salvar_matriz_2d(matrizes_d2n_k[i_corte],  f"{nome_script}_d2n_etapt_k_eta_{corte_str}.dat")
    salvar_matriz_2d(matrizes_d2n_d[i_corte],  f"{nome_script}_d2n_etapt_d_eta_{corte_str}.dat")
    salvar_matriz_2d(matrizes_d2n_ds[i_corte], f"{nome_script}_d2n_etapt_ds_eta_{corte_str}.dat")

    # Salva os valores evento-a-evento 
    c_events = os.path.join(output_dir, f"{nome_script}_events_eta_{corte_str}.dat")
    np.savetxt(
        c_events, 
        eventos_por_eta[i_corte], 
        fmt='%d %d %d %d',               
        header="N_ch N_D N_Ds N_K",     
        comments=''
    )

print(f"\nSimulação concluída!")
print(f"Total de eventos aceitos: {n_accepted}")
print(f"Arquivos salvos no diretório: {output_dir}")