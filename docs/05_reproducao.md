# Instalação e reprodução

[Voltar ao início](../README.md)

## Ambiente

A distribuição usa Python 3.11, NumPy 1.26.0 e SciPy 1.16.3. As avaliações AC e os refinamentos LP utilizam essas dependências. A validação local da distribuição foi realizada no Windows. Os oito motores nativos e o híbrido sem IA funcionam sem serviço externo.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install .
.venv/Scripts/python -m mestrado_grid methods
.venv/Scripts/python -m mestrado_grid verify
```

Também é possível instalar o arquivo `.whl` da distribuição com `python -m pip install caminho/do/arquivo.whl`. A execução pode ocorrer fora da pasta do código-fonte.

## Exemplo independente

```powershell
mestrado-grid prepare-demo --output exemplo --beta 0.5 --seed 96101
mestrado-grid run --method PSO --study exemplo/study.json --population exemplo/population.npy --output teste_PSO --budget 88 --seed 96101 --regime normal --case-kind synthetic
```

Substitua `PSO` por `HyDE`, `GA`, `ACO`, `Vortex`, `QPSO`, `SHADE`, `L-SHADE`, `HYBRID` ou `HYBRID_AI`. Use uma pasta de saída diferente em cada execução. O exemplo é uma rede sintética de três barras, com catálogo de linhas, geração, BESS e controles. Há oito ciclos diários, 32 estados e 96 casos AC por avaliação completa. Ele testa a integração do software; os resultados científicos da rede de 180 barras estão em [Resultados](06_resultados.md).

## Entradas de pesquisa

- `study.json`: objeto com `data` e `electrical`, contendo rede, catálogo, perfis, probabilidades, custos e limites.
- `population.npy`: oito vetores normalizados, no formato produzido por `ArrayCodec` para a mesma entrada.
- Semente, método e orçamento são parâmetros explícitos da execução.
- O executor de referência aceita 96 casos AC, α = 0,9 e β entre 0 e 1 em passos de 0,1, lidos da entrada.

`--regime normal` ou `--regime extremo` identifica os perfis já presentes na entrada; esse argumento não transforma os cenários nem modifica probabilidades. `--starting-point` registra a origem da população fornecida. O exemplo abaixo utiliza arquivos do usuário:

```powershell
mestrado-grid run --method L-SHADE --study dados/study.json --population dados/population.npy --output resultados/LSHADE_96101 --budget 4104 --seed 96101 --regime extremo --starting-point MAT
```

Os arquivos MAT, planilhas e entradas completas da rede de pesquisa são fornecidos separadamente. O caso sintético é gerado pelo próprio pacote.

## Híbrido com IA

`HYBRID_AI` exige o executável `codex` no PATH e autenticação válida. Utiliza inferência real, com observações do próprio ensaio, resposta estruturada e registro de requisições, decisões, uso e tempo de espera. Cada requisição tem limite de espera de 240 segundos. O supervisor não recebe ferramentas para acessar arquivos, executar comandos ou consultar outros ensaios. Suas decisões são validadas antes da aplicação; a avaliação física continua responsável pela aceitação dos candidatos.

O serviço pode consumir cota. Ausência de autenticação ou falha de inferência é registrada; não é substituída silenciosamente pelo método sem IA. A semente controla a busca local; o serviço de IA não é determinístico pela semente. Referência da integração: [execução não interativa do Codex](https://learn.chatgpt.com/docs/non-interactive-mode).

## Orçamento e saídas

`--budget` inclui os oito membros iniciais e todas as avaliações completas usadas durante a busca. Repetições também são debitadas. Há uma auditoria final independente adicional. Os métodos não encerram antecipadamente por atingir um valor financeiro.

Em cada saída:

| Arquivo/pasta | Conteúdo |
|---|---|
| `inputs/` e `job.json` | Cópias das entradas e parâmetros |
| `worker.stdout.log`, `worker.stderr.log` | Registros do processo |
| `result/status.json` | Estado e contagem de avaliações |
| `result/trajectory.jsonl` | Evolução do melhor candidato |
| `result/best_candidate.json` | Plano e operação finais |
| `result/validation.json.gz` | Avaliação AC independente completa |
| `result/summary.json` | FO, componentes, viabilidade, chamadas, tempos e uso da IA |
| `result/ai_review/` | Requisições e decisões, quando aplicável |

A conclusão do orçamento não implica viabilidade. Consulte o estado final, os 96 casos e as violações. A pasta de saída deve ser nova, evitando sobrescrita de evidências.

## Verificações

```powershell
python -m pip install ".[test]"
python -m pytest tests -q
python -B tools/verificar_documentacao.py
```

O primeiro teste verifica contratos do executor. O verificador documental confere links, hashes e evidências armazenadas. Os ensaios físicos dos dez métodos e a inferência real da IA são descritos em [Validação da distribuição](09_validacao.md).
