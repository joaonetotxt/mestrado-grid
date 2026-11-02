# Protocolo experimental de referência

[Voltar ao início](../README.md)

A comparação `comparacao_dez_metodos_pareada_20261020` é distinta das campanhas históricas que paravam ao atingir 25 ou 30 milhões. Sua regra é **orçamento fixo**, sem encerramento antecipado por meta financeira.

## Desenho

| Item | Valor |
|---|---|
| Métodos | PSO, HyDE, GA, ACO, Vortex, QPSO, SHADE, L-SHADE, HYBRID e HYBRID_AI |
| Partidas | População de origem MAT; população da heurística auxiliar |
| Sementes | 96101, 96109, 96121, 96137, 96149 |
| Risco | α = 0,9; β = 0 e 0,5 |
| Cenários | Regime com extremos |
| População inicial | Oito membros |
| Orçamento por ensaio | 4.104 avaliações completas: 8 de inicialização + 4.096 de busca |
| Auditoria final | Uma avaliação independente adicional |
| Cache de candidatos/casos | Desativado |
| Total planejado | 10 métodos × 2 partidas × 2 classes × 5 sementes = 200 ensaios |

O [protocolo preservado](../evidencias/protocolo_comparacao.json) contém os parâmetros e declarações originais. O total planejado não equivale a total concluído.

## Partidas e restrições

Dentro de cada tipo de partida, todos os métodos e sementes recebem a mesma população congelada de oito candidatos. Eventuais cópias na população heurística são declaradas. Não se usam finalistas otimizados das campanhas como se fossem partidas novas.

As restrições oficiais, inclusive SAIDI/SAIFI, valem desde a inicialização. A partida MAT pode ser não viável. Melhorar a viabilidade pode elevar a FO em relação a uma partida que violava metas.

Todas as opções do catálogo permanecem elegíveis, respeitados os vínculos físicos e temporais. Não há troca de candidatos entre braços. O custo anterior da heurística que construiu a segunda partida é excluído igualmente dessa comparação condicional e deve ser relatado à parte em uma análise de custo total.

## Contabilidade

Uma chamada completa ao avaliador é debitada mesmo quando o candidato se repete. Inicialização, retestes de contexto, diversidade, filhos, controles pareados e revalidações usadas na decisão consomem orçamento. A auditoria final independente é adicional.

São contabilizados separadamente:

- chamadas debitadas e casos AC efetivamente processados;
- tempo de parede e CPU local;
- número e tempo de soluções LP;
- requisições, uso e espera da IA;
- auditoria final e trabalho de preparação fora do ensaio.

Orçamento igual de avaliações **não significa** igualdade de tempo, computação remota ou custo monetário. Execuções concorrentes também influenciam tempo de parede.

## O que constitui um resultado final

Verificar conjuntamente `summary.json`, candidato salvo e `validation.json.gz`: conclusão, mesma identidade de candidato/plano, viabilidade, 96/96 casos esperados, todas as violações oficiais iguais a zero e conciliação financeira. “Zero” nos relatórios deve vir do valor armazenado, não de arredondamento visual.

`candidate_id` inclui operação; `plan_id` identifica a estrutura do plano. Planos iguais podem ter despachos e custos diferentes. Cada resultado precisa ainda de identificação da entrada, população, código e política financeira.

## Registro das execuções

Cada ensaio mantém a configuração, as identidades de código e entrada, a trajetória de busca e a auditoria final. A interface desta distribuição executa uma busca nova com orçamento finito, em diretório exclusivo; o histórico de qualquer recuperação pertence à execução que o originou.

## Comparação e limites das conclusões

Comparar primeiro dentro de **partida × β × semente**. Para múltiplas sementes, informar taxa de viabilidade e distribuição da FO dos finais viáveis; depois vitórias/empates/derrotas pareadas, convergência por avaliações e custo computacional real.

Não reunir classes de β ou partidas como réplicas equivalentes. Não transformar uma semente em evidência de superioridade geral. Ausência de melhora não prova ótimo global; elegibilidade não prova cobertura completa; auditoria nos cenários usados não substitui validação fora da amostra.
