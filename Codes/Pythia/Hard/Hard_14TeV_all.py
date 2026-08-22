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
pdg_D_mesons = [411, 431]       # D+, Ds+
pdg_K_mesons = [321]            # Kaons
bin_width = 0.1
nEvents = 10000

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
pythia.readString("PartonLevel:ISR = on") 
pythia.readString("PartonLevel:MPI = on")
pythia.readString("PartonLevel:FSR = on") 
pythia.init()

# =================================================================
# 3. Criar os Histogramas Nativos e Matrizes NumPy 2D
# =================================================================
h_mult_ch, h_mult_k, h_mult_d = [], [], []
h_eta_ch,  h_eta_k,  h_eta_d  = [], [], []
h_pt_ch,   h_pt_k,   h_pt_d   = [], [], []

# Eixos 2D globais (eta de -5.0 a 5.0; pT de 0.0 a 150.0)
eta_edges = np.arange(-5.0, 5.1, 0.1) 
pt_edges = np.arange(0.0, 150.1, 0.1)  

# Matrizes principais 2D para cada espécie
matriz_2d_ch = np.zeros((len(eta_edges)-1, len(pt_edges)-1))
matriz_2d_k  = np.zeros((len(eta_edges)-1, len(pt_edges)-1))
matriz_2d_d  = np.zeros((len(eta_edges)-1, len(pt_edges)-1))

for corte in cortes_eta:
    n_bins_eta = int(round(2.0 * corte / bin_width))
    pt_max_hist = 150.0  
    n_bins_pt  = int(round(pt_max_hist / bin_width))  
    
    # dN/deta
    h_eta_ch.append(pythia8.Hist(f"dN/deta Carregadas |eta|<{corte}", n_bins_eta, -corte, corte))
    h_eta_k.append(pythia8.Hist(f"dN/deta Kaons |eta|<{corte}", n_bins_eta, -corte, corte))
    h_eta_d.append(pythia8.Hist(f"dN/deta Mesons D |eta|<{corte}", n_bins_eta, -corte, corte))
    
    # dN/dpt
    h_pt_ch.append(pythia8.Hist(f"pT Carregadas |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))
    h_pt_k.append(pythia8.Hist(f"pT Kaons |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))
    h_pt_d.append(pythia8.Hist(f"pT Mesons D |eta|<{corte}", n_bins_pt, 0.0, pt_max_hist))

    # Multiplicidade
    h_mult_ch.append(pythia8.Hist(f"Mult Carregadas |eta|<{corte}", 400, -0.5, 399.5)) 
    h_mult_k.append(pythia8.Hist(f"Mult Kaons |eta|<{corte}", 100, -0.5, 99.5))
    h_mult_d.append(pythia8.Hist(f"Mult Mesons D |eta|<{corte}", 50, -0.5, 49.5))

n_accepted = 0

# =================================================================
# 4. Loop de Eventos
# =================================================================
for iEvt in range(nEvents):
    if not pythia.next():
        continue
    n_accepted += 1
    
    # Listas temporárias para preencher as matrizes 2D deste evento
    eta_evento_ch, pt_evento_ch = [], []
    eta_evento_k,  pt_evento_k  = [], []
    eta_evento_d,  pt_evento_d  = [], []

    n_ch_evento = [0] * len(cortes_eta)
    n_K_evento  = [0] * len(cortes_eta)
    n_D_evento  = [0] * len(cortes_eta)

    for i in range(pythia.event.size()):
        p = pythia.event[i]
        
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
        
        is_charged = (p.isCharged() and p.isFinal())
        is_kaon    = (pdg_abs in pdg_K_mesons and last_copy)
        is_meson_D = (pdg_abs in pdg_D_mesons and last_copy)

        if not (is_charged or is_kaon or is_meson_D):
            continue
            
        # Registra eta e pT nas listas 2D do evento (sem duplicações)
        if is_charged:
            eta_evento_ch.append(eta)
            pt_evento_ch.append(pt)
        if is_kaon:
            eta_evento_k.append(eta)
            pt_evento_k.append(pt)
        if is_meson_D:
            eta_evento_d.append(eta)
            pt_evento_d.append(pt)

        # Preenche os histogramas 1D para cada corte de eta
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

    # Preenche a multiplicidade do evento nas listas de histogramas 1D
    for i_corte in range(len(cortes_eta)):
        h_mult_ch[i_corte].fill(n_ch_evento[i_corte])
        h_mult_k[i_corte].fill(n_K_evento[i_corte])
        h_mult_d[i_corte].fill(n_D_evento[i_corte])
        
    # Converte as listas do evento atual em histogramas 2D e soma às matrizes globais
    if len(eta_evento_ch) > 0:
        h2d, _, _ = np.histogram2d(eta_evento_ch, pt_evento_ch, bins=[eta_edges, pt_edges])
        matriz_2d_ch += h2d
        
    if len(eta_evento_k) > 0:
        h2d, _, _ = np.histogram2d(eta_evento_k, pt_evento_k, bins=[eta_edges, pt_edges])
        matriz_2d_k += h2d
        
    if len(eta_evento_d) > 0:
        h2d, _, _ = np.histogram2d(eta_evento_d, pt_evento_d, bins=[eta_edges, pt_edges])
        matriz_2d_d += h2d

    if iEvt % 10000 == 0:
        print(f"Evento {iEvt} / {nEvents} processado.")

pythia.stat()

# =================================================================
# 5. Gravação dos Resultados
# =================================================================
for i_corte, corte in enumerate(cortes_eta):
    corte_str = f"{corte:.1f}".replace(".", "p")

    # Caminhos 1D
    c_mult_ch = os.path.join(output_dir, f"{nome_script}_mult_ch_eta_{corte_str}.dat")
    c_mult_k  = os.path.join(output_dir, f"{nome_script}_mult_k_eta_{corte_str}.dat")
    c_mult_d  = os.path.join(output_dir, f"{nome_script}_mult_d_eta_{corte_str}.dat")

    c_dndeta_ch = os.path.join(output_dir, f"{nome_script}_dndeta_ch_eta_{corte_str}.dat")
    c_dndeta_k  = os.path.join(output_dir, f"{nome_script}_dndeta_k_eta_{corte_str}.dat")
    c_dndeta_d  = os.path.join(output_dir, f"{nome_script}_dndeta_d_eta_{corte_str}.dat")
    
    c_pt_ch = os.path.join(output_dir, f"{nome_script}_pt_ch_eta_{corte_str}.dat")
    c_pt_k  = os.path.join(output_dir, f"{nome_script}_pt_k_eta_{corte_str}.dat")
    c_pt_d  = os.path.join(output_dir, f"{nome_script}_pt_d_eta_{corte_str}.dat")

    # Gravação via método .table() do Pythia
    h_mult_ch[i_corte].table(c_mult_ch)
    h_mult_k[i_corte].table(c_mult_k)
    h_mult_d[i_corte].table(c_mult_d)

    h_eta_ch[i_corte].table(c_dndeta_ch)
    h_eta_k[i_corte].table(c_dndeta_k)
    h_eta_d[i_corte].table(c_dndeta_d)
    
    h_pt_ch[i_corte].table(c_pt_ch)
    h_pt_k[i_corte].table(c_pt_k)
    h_pt_d[i_corte].table(c_pt_d)

# Gravação das Matrizes 2D globais via NumPy
cabecalho_2d = "Matriz 2D: Linhas = eta (-5.0 a 5.0, bins 0.1), Colunas = pT (0.0 a 150.0, bins 0.1)"

np.savetxt(os.path.join(output_dir, f"{nome_script}_matriz2D_ch.dat"), matriz_2d_ch, fmt='%g', header=cabecalho_2d)
np.savetxt(os.path.join(output_dir, f"{nome_script}_matriz2D_k.dat"), matriz_2d_k, fmt='%g', header=cabecalho_2d)
np.savetxt(os.path.join(output_dir, f"{nome_script}_matriz2D_d.dat"), matriz_2d_d, fmt='%g', header=cabecalho_2d)

print(f"Processamento concluído com sucesso! Salvo na subpasta: {output_dir}")