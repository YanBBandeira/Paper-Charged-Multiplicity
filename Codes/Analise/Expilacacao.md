A ordenação por percentil utilizada no seu script (e padronizada em colaborações como o **ALICE** no CERN) é uma técnica fundamental em física de colisões de íons pesados e próton-próton. O objetivo principal é classificar os eventos simulados ou colhidos com base na sua multiplicidade de partículas carregadas ($N_{\text{ch}}$), permitindo correlacionar a atividade global do evento com processos "duros" (neste caso, a produção de mésons $D_s$).

Abaixo está o detalhamento passo a passo de como essa ordenação ocorre e como o código a implementa:

---

## 1. O Princípio Físico da Centralidade e Percentis

Em colisões de alta energia, a multiplicidade de partículas carregadas ($N_{\text{ch}}$) medida em uma certa região de pseudorapidez ($\vert{}\eta\vert{}$) serve como um "cronômetro" ou indicador da severidade da colisão:

* **Eventos com alta multiplicidade** (muitas partículas geradas) correspondem a colisões centrais ou de alta densidade de energia.
* **Eventos com baixa multiplicidade** correspondem a colisões periféricas ou de baixa atividade.

As classes do ALICE (`classes_alice`) dividem o total de eventos acumulados em percentis cumulativos da seção de choque total, variando de $0\%$ (eventos mais extremos/multitudinários) até $100\%$ (eventos com menor multiplicidade). Por exemplo:

* **Classe I:** Os $0{,}95\%$ eventos com maior multiplicidade ($0.0\%$ a $0.95\%$).
* **Classe X:** Os eventos mais distantes da média de alta atividade, cobrindo a faixa de $68\%$ a $100\%$.

---

## 2. Como a Ordenação Ocorre no Código (Passo a Passo)

O processo lógico implementado nas funções `obter_norm_soft` e `obter_norm_hard` segue estas etapas:

### A. Ordenação Decrescente (`np.argsort`)

O código pega o array de multiplicidades de partículas carregadas (`n_ch_soft`) e o ordena de forma decrescente:

```python
sort_idx = np.argsort(-n_ch_soft)
n_ch_sorted = n_ch_soft[sort_idx]

```

* **Por que o sinal negativo (`-`)?** O `argsort` do NumPy ordena por padrão do menor para o maior (crescente). O sinal de menos inverte os valores, garantindo que o evento com o maior número de partículas fique na posição `0` do array ordenado, descendo até o evento com menor multiplicidade no final.

### B. Mapeamento de Percentis para Índices de Matriz

Para cada classe do ALICE (definida por um percentual inferior `p_inf_pct` e superior `p_sup_pct`), o script traduz a porcentagem em índices numéricos inteiros do array de eventos:

```python
i_start = int(np.round((p_inf_pct / 100.0) * n_evts))
i_end = max(i_start + 1, int(np.round((p_sup_pct / 100.0) * n_evts)))

```

* **Exemplo prático:** Se você tiver $100.000$ eventos (`n_evts`), a Classe I (de $0\%$ a $0{,}95\%$) pegará do índice `0` até o índice `950` do array ordenado.

### C. Fatiamento (*Slicing*) e Cálculo das Médias Locais

Com os índices de início e fim definidos, o script isola a fatia (*slice*) correspondente àquela classe de percentil tanto para o setor macio quanto para o setor duro:

```python
faixa = n_ch_sorted[i_start:i_end]
m_local = np.mean(faixa)  # Média de partículas carregadas na classe

```

No caso do setor duro (`obter_norm_hard`), o script usa a mesma ordenação de multiplicidade (`n_ch_hard` ordenado) para fatiar o array correspondente aos mésons $D_s$ (`n_ds_hard`), calculando a média local de partículas charm (`m_local_ds`) para aquele mesmo percentil de multiplicidade.

### D. Normalização Final

Por fim, os valores absolutos obtidos em cada classe são divididos pela média global de todo o conjunto de dados (`media_glob` e `media_glob_ds`):

$$\text{Norm} = \frac{\langle N \rangle_{\text{local}}}{\langle N \rangle_{\text{global}}}$$

Isso gera as coordenadas $(X, Y)$ normalizadas que formam os pontos dos seus gráficos de correlação. Se o comportamento fosse puramente linear (escala linear perfeita), os pontos seguiriam a diagonal tracejada ($y = x$), permitindo observar desvios físicos interessantes na produção de quarks charm em função da atividade subjacente do evento.