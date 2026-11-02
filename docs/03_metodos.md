# Métodos de busca

[Voltar ao início](../README.md)

## Elementos comuns

Cada método recebe a população definida pelo protocolo e propõe candidatos no espaço de decisões. A decodificação produz um plano de investimentos, topologias por estágio e operação temporal. Os reparos de representação e a agenda de blocos são adaptações declaradas ao problema.

O comparador prioriza viabilidade; entre candidatos não viáveis utiliza violações normalizadas; entre viáveis utiliza a FO selecionada. Empates, retestes de contexto e orçamento seguem o executor congelado. Nenhum método pode escolher outro cálculo financeiro para melhorar artificialmente seu resultado.

As referências dos algoritmos estão em [Referências](08_referencias.md). Este projeto desenvolve sua própria integração dos operadores com a representação da rede, a agenda de recursos e o protocolo de avaliação.

## Oito metaheurísticas

| Motor | Mecanismo central | Característica da implementação |
|---|---|---|
| PSO | Velocidade, memória individual e melhor posição coletiva | Limites, canonicalização e memória das posições efetivamente avaliadas |
| HyDE | Evolução diferencial com adaptação de parâmetros e perturbação do melhor | Variante HyDE |
| GA | Seleção, recombinação e mutação | Codificação mista e operador de comparação com restrições |
| ACO | Arquivo de soluções e amostragem contínua | Variante ACO_R no espaço de decisões da rede |
| Vortex | Amostragem ao redor de um centro com raio variável | Agenda de raio, blocos e orçamento executados |
| QPSO | Movimento probabilístico em torno de atratores | Tratamento dos limites e memórias na representação do projeto |
| SHADE | Evolução diferencial com memória do sucesso de parâmetros | Comparador com restrições e atualização das memórias |
| L-SHADE | SHADE com redução linear de população | População inicial 8, mínimo 4; relógio global de avaliações debitadas |

Os operadores determinam os valores propostos. A agenda compartilhada decide quais grupos de coordenadas recebem atenção: topologia, capacidades, operação P/Q, atendimento, armazenamento e outros recursos. Essa integração incorpora conhecimento do problema ao método avaliado.

Persistir memória de um motor não autoriza reutilizar fitness de um contexto físico alterado. Retestes necessários contam no orçamento. No L-SHADE, a redução populacional acompanha o orçamento global, enquanto memórias e arquivo podem sobreviver a revisitas.

## Híbrido sem IA

`HYBRID` combina busca estruturada, modificações de ativos/topologia e refinamentos analíticos de operação. Inclui subproblemas de programação linear para propor ajustes de despacho. A proposta resultante passa pelo avaliador AC completo e pelas mesmas restrições do estudo.

A contagem e o tempo dos LPs são registrados separadamente. Não se deve chamar esse braço de metaheurística puramente nativa, nem equiparar custo de LP a uma avaliação AC completa.

## Híbrido com IA

`HYBRID_AI` usa o mesmo núcleo híbrido com um supervisor que recebe observações do próprio ensaio e revisa planos de busca. O plano pode orientar recursos, intervenções e sequência de experimentos dentro das ações implementadas.

A IA não avalia a física, não certifica viabilidade e não modifica a FO. As propostas são materializadas pelo executor; a aceitação depende da avaliação e do comparador oficiais. Uma decisão registrada não é evidência de ganho até que exista candidato avaliado e aceito.

No protocolo de referência, a inferência autônoma é realizada por Codex autenticado, sem ferramentas ou acesso direto a arquivos durante a decisão. São registrados requisições, uso, espera e falhas. A semente fixa a aleatoriedade algorítmica, mas não garante determinismo do serviço de IA.

O supervisor tem um conjunto de ações implementado; “com IA” não significa capacidade ilimitada de modificar o modelo ou acesso a resultados de outros braços. A indisponibilidade do serviço não deve ser substituída silenciosamente por uma decisão simulada de IA.

## Independência e atribuição de ganhos

Cada braço conserva candidatos e observações próprios. Populações iniciais iguais são um controle experimental, não uma transferência de melhorias entre métodos. O custo histórico de preparar a partida heurística não está incluído no orçamento condicional da comparação.

Separar ganhos por origem imediata — inicialização, reteste, proposta nativa, refinamento, troca de topologia ou plano da IA — é útil para diagnóstico. Essa classificação não isola causalidade. A contribuição exclusiva da IA exige controles e comparação pareada, não apenas um melhor finalista.

GVNS/VND, HyDE-DF e a extensão de ilhamento com BESS pertencem ao histórico de investigação. Eles não são braços adicionais do protocolo de dez métodos apresentado aqui.
