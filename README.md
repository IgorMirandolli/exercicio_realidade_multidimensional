# Realidade Multidimensional: ID3 vs C4.5

Este projeto resolve a atividade usando o problema de aprovacao de Pull Request.

## Como executar

```powershell
python -m pip install -r requirements.txt
python atividade_id3_c45.py
```

## O que o script faz

- Cria um dataset com 15 linhas, 5 atributos de entrada e 1 classe de saida.
- Monta um dicionario de dados explicando cada atributo.
- Aplica `LabelEncoder` em todos os atributos e no alvo.
- Treina `DecisionTreeClassifier(criterion="entropy")` para simular o ID3.
- Mostra profundidade, folhas, raiz, importancias e arvore textual.
- Usa `inverse_transform` para traduzir a predicao final para `S` ou `N`.
- Adiciona `ID_Evento` como atributo parasita.
- Compara Ganho de Informacao com Razao de Ganho.
- Simula um C4.5 didatico com `criterion="entropy"`, `max_depth=5` e `ccp_alpha=0.01`.
- Mostra por que poda sozinha nao substitui a Razao de Ganho quando existe um ID unico.

## Observacao importante

O `DecisionTreeClassifier` do scikit-learn usa CART, que cria arvores binarias.
Por isso, ele nao replica perfeitamente a arvore multi-ramo do ID3/C4.5
classico. Para deixar o duelo claro, o script tambem calcula manualmente:

- Ganho de Informacao, que favorece o ID unico.
- Razao de Ganho, que pune atributos com muitos valores diferentes.

Essa diferenca mostra por que um identificador unico pode fazer o ID3 decorar o
historico em vez de aprender um padrao util.

Na parte extra, o script imprime tres leituras:

- ID3 com `ID_Evento`, que tende a escolher o identificador.
- CART apenas podado, mostrando que `ccp_alpha` e `max_depth` nao implementam
  Gain Ratio por conta propria.
- C4.5 didatico, descartando `ID_Evento` pela leitura de Razao de Ganho antes de
  treinar a arvore podada.
