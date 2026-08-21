import os
import pythia8

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
pdg_K_mesons = [321]  # Kaons
bin_width = 0.1
nEvents = 10000#1000000

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
# 3. Criar os Histogramas Nativos em Listas (Estilo Procedural/Array)
# =================================================================
h_mult_ch, h_mult_k, h_mult_d = [], [], []
h_eta_ch,  h_eta_k,  h_eta_d  = [], [], []

for corte in cortes_eta:
    n_bins_eta = int(round(2.0 * corte / bin_width))
    
    # dN/deta
    h_eta_ch.append(pythia8.Hist(f"dN/deta Carregadas |eta|<{corte}", n_bins_eta, -corte, corte))
    h_eta_k.append(pythia8.Hist(f"dN/deta Kaons |eta|<{corte}", n_bins_eta, -corte, corte))
    h_eta_d.append(pythia8.Hist(f"dN/deta Mesons D |eta|<{corte}", n_bins_eta, -corte, corte))

    # h_mult (Eventos vs Multiplicidade N)
    # (Número de bins, limite inferior, limite superior)
    # Com limites deslocados em -0.5, o bin de N=0 vai de [-0.5 a 0.5], e o centro lido pelo Python será exatamente 0.0!
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

        # Lógica de última cópia
        
        is_charged = (p.isCharged() and p.isFinal())
        is_kaon    = (pdg_abs in pdg_K_mesons and last_copy)
        is_meson_D = (pdg_abs in pdg_D_mesons and last_copy)

        if not (is_charged or is_kaon or is_meson_D):
            continue

        for i, corte in enumerate(cortes_eta):
            if eta_abs < corte:
                if is_charged:
                    h_eta_ch[i].fill(eta)
                    n_ch_evento[i] += 1
                if is_kaon:
                    h_eta_k[i].fill(eta)
                    n_K_evento[i] += 1
                if is_meson_D:
                    h_eta_d[i].fill(eta)
                    n_D_evento[i] += 1

    # Preenche a multiplicidade do evento nas listas de histogramas
    for i in range(len(cortes_eta)):
        h_mult_ch[i].fill(n_ch_evento[i])
        h_mult_k[i].fill(n_K_evento[i])
        h_mult_d[i].fill(n_D_evento[i])

    if iEvt % 10000 == 0:
        print(f"Evento {iEvt} / {nEvents} processado.")

pythia.stat()

# =================================================================
# 5. Gravação dos Resultados Usando a Tabela Nativa do Pythia
# =================================================================
for i, corte in enumerate(cortes_eta):
    corte_str = f"{corte:.1f}".replace(".", "p")

    # Caminhos individuais para cada arquivo .dat
    c_mult_ch = os.path.join(output_dir, f"{nome_script}_mult_ch_eta_{corte_str}.dat")
    c_mult_k  = os.path.join(output_dir, f"{nome_script}_mult_k_eta_{corte_str}.dat")
    c_mult_d  = os.path.join(output_dir, f"{nome_script}_mult_d_eta_{corte_str}.dat")

    c_dndeta_ch = os.path.join(output_dir, f"{nome_script}_dndeta_ch_eta_{corte_str}.dat")
    c_dndeta_k  = os.path.join(output_dir, f"{nome_script}_dndeta_k_eta_{corte_str}.dat")
    c_dndeta_d  = os.path.join(output_dir, f"{nome_script}_dndeta_d_eta_{corte_str}.dat")

    # Gravação nativa via método .table() do Pythia
    h_mult_ch[i].table(c_mult_ch)
    h_mult_k[i].table(c_mult_k)
    h_mult_d[i].table(c_mult_d)

    h_eta_ch[i].table(c_dndeta_ch)
    h_eta_k[i].table(c_dndeta_k)
    h_eta_d[i].table(c_dndeta_d)

print(f"Processamento concluído com sucesso! Salvo na subpasta: {output_dir}")