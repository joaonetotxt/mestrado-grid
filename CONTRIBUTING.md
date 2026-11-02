# Como contribuir

Contribuições devem preservar a identificação da versão científica e incluir verificações proporcionais à mudança proposta.

## Relatar um problema

Informar versão/commit ou hash, entrada e protocolo, método, partida, semente, α/β, orçamento, comportamento esperado e comportamento observado. Para um resultado, incluir `candidate_id`, `plan_id` e um exemplo mínimo. Não anexar credenciais ou dados cuja distribuição não esteja autorizada.

## Mudanças científicas

- Alterações de física, restrições, cenário, tarifa ou FO exigem nova identificação de versão e justificativa.
- Melhorias de busca devem ser comparadas em testes isolados com partidas e orçamentos identificados.
- Preservar operadores nativos nos braços definidos como nativos; refinadores LP ou IA precisam ser declarados.
- Não sobrescrever fontes congeladas, entradas ou resultados históricos.
- Retomadas devem declarar exatamente o que foi recuperado e o que foi reinicializado.
- Atualizar a documentação e os testes pertinentes ao comportamento modificado.

## Mudanças documentais

Conferir nomes de políticas, unidades, referência temporal dos custos, links e origem das tabelas. Usar valores exatos nas evidências e arredondar apenas a apresentação. Executar:

```powershell
python -B tools/verificar_documentacao.py
```

Os hashes do manifesto devem ser atualizados de forma explícita após uma revisão aprovada; não devem ser modificados silenciosamente pelo verificador.
