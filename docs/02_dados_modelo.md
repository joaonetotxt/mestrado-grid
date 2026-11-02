# Dados, física e economia

[Voltar ao início](../README.md)

## Rede de origem e conciliação

A estrutura vem do `caseStudyData.mat` fornecido ao projeto. O estudo utiliza 180 barras, 179 ligações da árvore inicial e 22 alternativas, totalizando 201 trechos no catálogo. “Partida MAT” identifica a origem da topologia, não a utilização do MAT bruto sem as conciliações posteriores.

A camada de cargas representa as **90 cargas da planilha**, mantendo o mapeamento de localização assumido pelo projeto. Entradas residuais da base conciliada foram zeradas. O modelo adota a correspondência entre as localizações da planilha e as barras do MAT como hipótese de entrada.

O inventário do estudo de referência contém 44 unidades PV, 55 eólicas e três biomassas, além da subestação; há operação EV em cinco barras.

## Potência reativa

O Q original é preservado como referência. Na variante conciliada utilizada nas comparações posteriores:

$$Q_{novo,i,t}=P_{novo,i,t}\frac{Q_{original,i,t}}{P_{original,i,t}}.$$

Isso preserva o fator de potência individual correspondente ao perfil original. O fator de potência pode ser calculado por $FP=P/\sqrt{P^2+Q^2}$; não é necessário sortear um FP para obter o caso-base. Preservar a razão Q/P é uma hipótese sobre as novas cargas. Sensibilidades com outros fatores de potência devem ter entradas e resultados separados.

A auditoria da conciliação registra P original positivo nas 90 cargas ativas e resíduos P/Q exatamente zerados. Não se deve transformar esse tratamento em uma regra genérica de divisão por zero para outras bases.

## Tempo e cenários

| Elemento | Convenção da comparação de referência |
|---|---|
| Estágios de investimento | Anos 0, 10 e 20 |
| Horizonte econômico | 30 anos |
| Estados | 32, com cenários extremos no protocolo comparativo |
| Casos AC por avaliação completa | 3 estágios × 32 estados = 96 |
| Representação diária | Oito dias representativos, com ciclos de 24 horas |
| Pesos anuais | Probabilidades e horas anuais, com referência de 8.760 h |
| Armazenamento | Ciclo completo dentro de cada dia; sem transferência de SOC entre dias representativos |

Os cenários são construídos para este estudo. O regime de cenários e o valor de β identificam cada experimento.

## Decisões e restrições

O catálogo permite topologia, linhas, geração planejada, despacho P/Q, atendimento, resposta da demanda, EV, BESS, capacitores, reguladores e adequações físicas. A presença de uma opção no catálogo não comprova que todas as combinações de local, tamanho e operação foram visitadas.

A avaliação verifica radialidade/conectividade e consistência das compras; balanços e limites elétricos; tensão; corrente e potência aparente; subestação; geração; DR/EV; SOC e potência BESS; instalação dos controles; chaveamentos; SAIDI/SAIFI. As tolerâncias pertencem à política do avaliador e devem acompanhar a versão publicada.

`feasible`, `violated` e `inconclusive` são estados distintos. Convergência em 96 casos AC não basta para aprovar restrições temporais ou de confiabilidade. Falha numérica também não prova inviabilidade física.

SAIDI/SAIFI são indicadores calculados segundo a representação topológica de falhas e reparos do modelo. Corte programado de carga é uma variável operacional distinta. Não é correto interpretar automaticamente menor SAIDI como menor corte operacional.

## BESS e propriedade

BESS permanece elegível para instalação e operação P/Q, com balanço energético, eficiência e ciclo SOC completo. A extensão experimental de atendimento em ilha **não integra o modelo de referência**, conforme decisão de escopo registrada no projeto.

O modelo considera BESS como ativo do operador, com os custos de investimento e operação registrados nessa perspectiva. A instalação e o despacho são decisões da otimização, condicionadas ao benefício econômico e às restrições do sistema.

## Base econômica de Castro

A referência econômica é [Castro et al. (2024)](08_referencias.md), com [Canizes et al. (2019)](08_referencias.md) como base de planejamento e armazenamento. A formulação econômica adotada neste projeto é:

$$J=PC1_{VP}+(1-\beta)\mathbb{E}[C_{anual}]+\beta CVaR_\alpha(C_{anual}).$$

- `PC1_VP`: investimentos e manutenção fixos em valor presente.
- `C_anual`: operação anual equivalente ponderada entre os estágios.
- `α`: nível de confiança; na comparação, 0,9.
- `β`: peso entre expectativa e CVaR; na comparação, 0 ou 0,5.

Com β = 1, apenas o CVaR operacional entra na parcela variável; com β = 0, entra a expectativa. Não se deve escrever `PC1 + E + β·CVaR` como se fosse a mesma função.

O CVaR usa os custos condicionais e as probabilidades dos estados, permitindo massa fracionária no limite da cauda. Sua base teórica é [Rockafellar e Uryasev (2002)](08_referencias.md). As probabilidades não devem ser substituídas por uma média simples dos cenários.

Os custos incluem os componentes aplicáveis de rede, recursos, geração, emissões, contratos, DR/EV, corte, curtailment, perdas AC e proxy FOR. Valores pagos pelo operador e custos dos produtores não devem ser somados duas vezes.

`selection_objective` identifica a FO usada na busca. VP30 e EAC permanecem separados; a combinação Castro-aligned não deve ser rotulada como VP30 ou como EAC. A taxa e os demais coeficientes econômicos são os congelados nas entradas, sem atualização automática para preços atuais.

## Leitura das energias

Apresentar separadamente: carga cortada, curtailment renovável conectado e indisponibilidade renovável. Reduzir capacidade renovável planejada pode diminuir o curtailment absoluto; isso não demonstra, por si só, maior aproveitamento da geração disponível.
