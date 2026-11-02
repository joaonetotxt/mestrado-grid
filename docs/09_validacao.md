# Validação da distribuição 0.1.0

[Voltar ao início](../README.md)

O pacote foi instalado a partir de um arquivo wheel em um ambiente virtual novo, sem acesso aos pacotes do ambiente das campanhas. Os comandos foram executados fora da árvore de código. Ambiente verificado: Windows, Python 3.11.9, NumPy 1.26.0 e SciPy 1.16.3.

## Testes automatizados

**38 testes aprovados**, cobrindo:

- contagem de avaliações, repetições e separação da auditoria;
- seleção pela FO efetiva e prioridade das restrições nos motores;
- composição financeira independente e CVaR para β = 0, 0,5 e 1;
- comparação entre dois solvers AC, perdas e balanço nos terminais;
- capacitores, instalação temporal e controles físicos;
- balanço diário, limites de potência e conservação de energia dos BESS;
- proteção de saídas existentes e validação da interface instalada.

## Integração dos dez métodos

Foi utilizado o mesmo caso sintético de três barras, a mesma população de oito candidatos, semente 96101, α = 0,9 e β = 0,5. Cada método recebeu 88 avaliações completas, incluindo inicialização, e uma auditoria final adicional.

| Método | Teste | Avaliações | Auditoria final | Propostas nativas | Soluções LP |
|---|---|---:|---:|---:|---:|
| PSO | Aprovado | 88 | 96/96 | 56 | 0 |
| HyDE | Aprovado | 88 | 96/96 | 56 | 0 |
| GA | Aprovado | 88 | 96/96 | 56 | 0 |
| ACO | Aprovado | 88 | 96/96 | 56 | 0 |
| Vortex | Aprovado | 88 | 96/96 | 56 | 0 |
| QPSO | Aprovado | 88 | 96/96 | 56 | 0 |
| SHADE | Aprovado | 88 | 96/96 | 56 | 0 |
| L-SHADE | Aprovado | 88 | 96/96 | 56 | 0 |
| HYBRID | Aprovado | 88 | 96/96 | — | 53 |
| HYBRID_AI | Aprovado | 88 | 96/96 | — | 35 |

Todos os finais foram viáveis nos 96 casos, com violações oficiais armazenadas iguais a zero e conciliação financeira aprovada. As oito metaheurísticas produziram descendentes nativos; os híbridos executaram os refinamentos LP. O híbrido com IA realizou **duas requisições reais**, com respostas aceitas, sem uso de ferramentas pelo supervisor.

A contabilidade total foi de **880 avaliações de busca / 84.480 casos AC**, mais **10 auditorias / 960 casos AC**. Esses ensaios verificam instalação e integração; a comparação científica da rede de pesquisa está em [Resultados de referência](06_resultados.md).

## Reavaliação da rede de pesquisa

Uma solução histórica do híbrido com IA foi reavaliada com o código instalado na rede de 180 barras. O candidato e o plano mantiveram suas identidades, com viabilidade em 96/96 casos e todas as violações oficiais iguais a zero. A FO e seus componentes financeiros coincidiram dentro da tolerância absoluta de 0,00001 u.m. usada pelo protocolo.

## Identidade científica

Os **166 arquivos de código científico** conservam os hashes das fontes de referência. O arquivo de parâmetros contém os mesmos parâmetros de motores e busca, em formato portátil. A interface de distribuição configura caminhos, entradas e saídas; avaliador físico, operadores e contabilidade financeira permanecem preservados.

[Registro numérico dos testes e identidades](../evidencias/validacao_distribuicao.json).

Para executar novamente o caso de integração com a distribuição instalada:

```powershell
python tools/smoke_methods.py --output ensaio_integracao
python tools/smoke_methods.py --output ensaio_integracao_ia --methods HYBRID_AI --with-ai
```

O segundo comando utiliza o serviço de IA autenticado e consome cota.
