import os
import numpy as np

# Configurações iguais às da simulação
nome_script = "Hard_14TeV_all"
subpasta_nome = "all"
output_dir = os.path.join("..", "Dados", "Hard", subpasta_nome)

cortes_eta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
especies = ["ch", "k", "d"]
bin_width = 0.1

print(f"{'Corte':<6} | {'Esp':<4} | {'Soma P(N)':<10} | {'<N>':<8} | {'Integral dN/deta':<18} | {'<dN/deta> Global'}")
print("-" * 75)

# Loop Procedural: Para cada corte, lemos todas as espécies
for corte in cortes_eta:
    corte_str = f"{corte:.1f}".replace(".", "p")
    delta_eta = 2.0 * corte
    
    for esp in especies:
        # 1. Definir caminhos dos arquivos
        arq_mult = os.path.join(output_dir, f"{nome_script}_mult_{esp}_eta_{corte_str}.dat")
        arq_eta  = os.path.join(output_dir, f"{nome_script}_dndeta_{esp}_eta_{corte_str}.dat")
        
        # 2. Ler e processar Multiplicidade (Agora usando a coluna 1)
        dados_mult = np.loadtxt(arq_mult)
        N_valores = dados_mult[:, 0] # Eixo X: Multiplicidade
        N_eventos = dados_mult[:, 1] # Eixo Y: Contagem absoluta de eventos
        
        total_eventos = np.sum(N_eventos)
        
        # Prevenção contra divisão por zero caso o arquivo esteja vazio
        if total_eventos == 0:
            print(f"{corte:<6.1f} | {esp:<4} | {'0.0':<10} | {'0.0':<8} | {'0.0':<18} | 0.0")
            continue
            
        # Probabilidade e Média
        P_N = N_eventos / total_eventos
        media_N = np.sum(N_valores * P_N)
        
        # TESTE 1: A soma de P(N) deve ser 1.0
        soma_prob = np.sum(P_N)
        
        # 3. Ler e processar dN/deta (Agora usando a coluna 1)
        dados_eta = np.loadtxt(arq_eta)
        soma_particulas = dados_eta[:, 1] # Eixo Y: Total de partículas em cada bin
        
        # Distribuição normalizada por evento e por largura do bin
        dndeta_dist = soma_particulas / (total_eventos * bin_width)
        
        # TESTE 2: A integral de dN/deta (soma das densidades * largura do bin) deve ser igual a <N>
        integral_dndeta = np.sum(dndeta_dist * bin_width)
        
        # Densidade global (um número só para o intervalo todo)
        dndeta_global = media_N / delta_eta
        
        # 4. Imprimir resultados formatados
        print(f"{corte:<6.1f} | {esp:<4} | {soma_prob:<10.4f} | {media_N:<8.4f} | {integral_dndeta:<18.4f} | {dndeta_global:.4f}")

print("-" * 75)
print("Se 'Soma P(N)' for 1.0000 e '<N>' for igual à 'Integral dN/deta', sua normalização está perfeita!")