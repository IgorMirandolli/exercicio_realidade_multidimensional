from __future__ import annotations

from collections import Counter, defaultdict
from math import log2
from typing import Any

try:
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier, export_text
except ImportError as exc:
    raise SystemExit(
        "Dependencia ausente. Execute: python -m pip install -r requirements.txt"
    ) from exc


TARGET = "Aprovar"

ATRIBUTOS_5 = [
    "Tamanho_PR",
    "Tem_Testes",
    "Complexidade_Ciclo",
    "Autor_Senior",
    "Urgencia",
]

DICIONARIO_DADOS = {
    "Tamanho_PR": {
        "descricao": "Quantidade de mudancas no pull request.",
        "valores": {"P": "Pequeno", "M": "Medio", "G": "Grande"},
    },
    "Tem_Testes": {
        "descricao": "Indica se o PR possui testes automatizados.",
        "valores": {"S": "Sim", "N": "Nao"},
    },
    "Complexidade_Ciclo": {
        "descricao": "Complexidade do ciclo de revisao.",
        "valores": {"A": "Alta", "B": "Baixa"},
    },
    "Autor_Senior": {
        "descricao": "Indica se o autor tem senioridade alta.",
        "valores": {"S": "Sim", "N": "Nao"},
    },
    "Urgencia": {
        "descricao": "Indica se a entrega e urgente.",
        "valores": {"S": "Sim", "N": "Nao"},
    },
    "ID_Evento": {
        "descricao": "Identificador unico criado apenas para o experimento parasita.",
        "valores": "Numeros unicos de 1 a 15.",
    },
    TARGET: {
        "descricao": "Classe de saida: se o pull request deve ser aprovado.",
        "valores": {"S": "Aprovar", "N": "Nao aprovar"},
    },
}

DADOS = [
    {
        "Tamanho_PR": "P",
        "Tem_Testes": "S",
        "Complexidade_Ciclo": "B",
        "Autor_Senior": "S",
        "Urgencia": "N",
        TARGET: "S",
    },
    {
        "Tamanho_PR": "M",
        "Tem_Testes": "S",
        "Complexidade_Ciclo": "B",
        "Autor_Senior": "N",
        "Urgencia": "N",
        TARGET: "S",
    },
    {
        "Tamanho_PR": "P",
        "Tem_Testes": "S",
        "Complexidade_Ciclo": "A",
        "Autor_Senior": "S",
        "Urgencia": "S",
        TARGET: "S",
    },
    {
        "Tamanho_PR": "G",
        "Tem_Testes": "S",
        "Complexidade_Ciclo": "B",
        "Autor_Senior": "S",
        "Urgencia": "S",
        TARGET: "S",
    },
    {
        "Tamanho_PR": "M",
        "Tem_Testes": "S",
        "Complexidade_Ciclo": "B",
        "Autor_Senior": "S",
        "Urgencia": "N",
        TARGET: "S",
    },
    {
        "Tamanho_PR": "P",
        "Tem_Testes": "S",
        "Complexidade_Ciclo": "B",
        "Autor_Senior": "N",
        "Urgencia": "S",
        TARGET: "S",
    },
    {
        "Tamanho_PR": "G",
        "Tem_Testes": "S",
        "Complexidade_Ciclo": "A",
        "Autor_Senior": "S",
        "Urgencia": "N",
        TARGET: "S",
    },
    {
        "Tamanho_PR": "M",
        "Tem_Testes": "S",
        "Complexidade_Ciclo": "A",
        "Autor_Senior": "N",
        "Urgencia": "S",
        TARGET: "S",
    },
    {
        "Tamanho_PR": "G",
        "Tem_Testes": "S",
        "Complexidade_Ciclo": "A",
        "Autor_Senior": "N",
        "Urgencia": "N",
        TARGET: "N",
    },
    {
        "Tamanho_PR": "P",
        "Tem_Testes": "N",
        "Complexidade_Ciclo": "B",
        "Autor_Senior": "S",
        "Urgencia": "S",
        TARGET: "N",
    },
    {
        "Tamanho_PR": "M",
        "Tem_Testes": "N",
        "Complexidade_Ciclo": "B",
        "Autor_Senior": "S",
        "Urgencia": "N",
        TARGET: "N",
    },
    {
        "Tamanho_PR": "G",
        "Tem_Testes": "N",
        "Complexidade_Ciclo": "A",
        "Autor_Senior": "S",
        "Urgencia": "S",
        TARGET: "N",
    },
    {
        "Tamanho_PR": "P",
        "Tem_Testes": "N",
        "Complexidade_Ciclo": "A",
        "Autor_Senior": "N",
        "Urgencia": "S",
        TARGET: "N",
    },
    {
        "Tamanho_PR": "M",
        "Tem_Testes": "N",
        "Complexidade_Ciclo": "A",
        "Autor_Senior": "N",
        "Urgencia": "N",
        TARGET: "N",
    },
    {
        "Tamanho_PR": "G",
        "Tem_Testes": "N",
        "Complexidade_Ciclo": "B",
        "Autor_Senior": "N",
        "Urgencia": "N",
        TARGET: "N",
    },
]

DADOS_COM_ID = [
    {**linha, "ID_Evento": indice + 1} for indice, linha in enumerate(DADOS)
]


def formatar_valor(valor: Any) -> str:
    if isinstance(valor, float):
        return f"{valor:.3f}"
    return str(valor)


def formatar_tabela(cabecalho: list[str], linhas: list[list[Any]]) -> str:
    texto_linhas = [[formatar_valor(celula) for celula in linha] for linha in linhas]
    larguras = [
        max(len(cabecalho[i]), *(len(linha[i]) for linha in texto_linhas))
        for i in range(len(cabecalho))
    ]
    separador = "-+-".join("-" * largura for largura in larguras)
    topo = " | ".join(cabecalho[i].ljust(larguras[i]) for i in range(len(cabecalho)))
    corpo = [
        " | ".join(linha[i].ljust(larguras[i]) for i in range(len(cabecalho)))
        for linha in texto_linhas
    ]
    return "\n".join([topo, separador, *corpo])


def entropia(valores: list[Any]) -> float:
    total = len(valores)
    contagem = Counter(valores)
    return -sum((qtd / total) * log2(qtd / total) for qtd in contagem.values())


def particionar(dados: list[dict[str, Any]], atributo: str) -> dict[Any, list[dict[str, Any]]]:
    grupos: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for linha in dados:
        grupos[linha[atributo]].append(linha)
    return grupos


def ganho_informacao(dados: list[dict[str, Any]], atributo: str, target: str) -> float:
    total = len(dados)
    entropia_base = entropia([linha[target] for linha in dados])
    entropia_restante = 0.0

    for grupo in particionar(dados, atributo).values():
        peso = len(grupo) / total
        entropia_restante += peso * entropia([linha[target] for linha in grupo])

    return entropia_base - entropia_restante


def split_info(dados: list[dict[str, Any]], atributo: str) -> float:
    return entropia([linha[atributo] for linha in dados])


def razao_ganho(dados: list[dict[str, Any]], atributo: str, target: str) -> float:
    divisor = split_info(dados, atributo)
    if divisor == 0:
        return 0.0
    return ganho_informacao(dados, atributo, target) / divisor


def ranking_teorico(
    dados: list[dict[str, Any]], atributos: list[str], target: str
) -> list[list[Any]]:
    linhas = []
    for atributo in atributos:
        linhas.append(
            [
                atributo,
                ganho_informacao(dados, atributo, target),
                split_info(dados, atributo),
                razao_ganho(dados, atributo, target),
            ]
        )
    return sorted(linhas, key=lambda linha: linha[1], reverse=True)


def codificar(
    dados: list[dict[str, Any]], atributos: list[str], target: str
) -> tuple[list[list[int]], list[int], dict[str, LabelEncoder], LabelEncoder]:
    encoders: dict[str, LabelEncoder] = {}
    colunas_codificadas: dict[str, list[int]] = {}

    for atributo in atributos:
        encoder = LabelEncoder()
        valores = [linha[atributo] for linha in dados]
        colunas_codificadas[atributo] = encoder.fit_transform(valores).tolist()
        encoders[atributo] = encoder

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform([linha[target] for linha in dados]).tolist()
    x = [
        [colunas_codificadas[atributo][indice] for atributo in atributos]
        for indice in range(len(dados))
    ]

    return x, y, encoders, target_encoder


def imprimir_dicionario() -> None:
    print("\n=== Dicionario de dados ===")
    linhas = []
    for campo, info in DICIONARIO_DADOS.items():
        linhas.append([campo, info["descricao"], info["valores"]])
    print(formatar_tabela(["Campo", "Descricao", "Valores"], linhas))


def imprimir_mapeamento(
    encoders: dict[str, LabelEncoder], target_encoder: LabelEncoder
) -> None:
    print("\n=== Mapeamento criado pelo LabelEncoder ===")
    linhas = []
    for atributo, encoder in encoders.items():
        mapa = ", ".join(
            f"{classe} -> {codigo}" for codigo, classe in enumerate(encoder.classes_)
        )
        linhas.append([atributo, mapa])

    mapa_target = ", ".join(
        f"{classe} -> {codigo}"
        for codigo, classe in enumerate(target_encoder.classes_)
    )
    linhas.append([TARGET, mapa_target])
    print(formatar_tabela(["Campo", "Codificacao"], linhas))


def raiz_da_arvore(modelo: DecisionTreeClassifier, atributos: list[str]) -> str:
    indice = modelo.tree_.feature[0]
    if indice < 0:
        return "Folha unica"
    return atributos[indice]


def importancias(modelo: DecisionTreeClassifier, atributos: list[str]) -> str:
    pares = sorted(
        zip(atributos, modelo.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    return ", ".join(f"{nome}={valor:.3f}" for nome, valor in pares if valor > 0)


def avaliar_modelo(
    nome: str,
    modelo: DecisionTreeClassifier,
    x: list[list[int]],
    y: list[int],
    atributos: list[str],
) -> DecisionTreeClassifier:
    modelo.fit(x, y)
    acuracia_treino = accuracy_score(y, modelo.predict(x))
    loo = LeaveOneOut()
    acuracia_loo = cross_val_score(modelo, x, y, cv=loo).mean()

    print(f"\n=== {nome} ===")
    print(
        formatar_tabela(
            ["Metrica", "Valor"],
            [
                ["Acuracia no historico", acuracia_treino],
                ["Acuracia Leave-One-Out", acuracia_loo],
                ["Profundidade", modelo.get_depth()],
                ["Folhas", modelo.get_n_leaves()],
                ["Raiz", raiz_da_arvore(modelo, atributos)],
                ["Importancias", importancias(modelo, atributos) or "Todas zero"],
            ],
        )
    )
    print("\nArvore textual:")
    print(export_text(modelo, feature_names=atributos, decimals=0))

    if acuracia_treino == 1.0 and acuracia_loo < 0.80:
        print(
            "Leitura critica: ha sinal de memorizacao, pois o historico foi "
            "decorado melhor do que os casos deixados de fora."
        )
    if "ID_Evento" in atributos and modelo.get_depth() >= 4:
        print(
            "Leitura critica: com ID_Evento presente, a profundidade alta e "
            "um alerta de que a arvore pode estar usando identificadores."
        )

    return modelo


def predicao_legivel(
    modelo: DecisionTreeClassifier,
    encoders: dict[str, LabelEncoder],
    target_encoder: LabelEncoder,
    atributos: list[str],
    novo_caso: dict[str, Any],
) -> str:
    x_novo = [[encoders[atributo].transform([novo_caso[atributo]])[0] for atributo in atributos]]
    predicao_codificada = modelo.predict(x_novo)
    return target_encoder.inverse_transform(predicao_codificada)[0]


def imprimir_ranking(dados: list[dict[str, Any]], atributos: list[str], titulo: str) -> None:
    print(f"\n=== {titulo} ===")
    print(
        formatar_tabela(
            ["Atributo", "Ganho_Info_ID3", "Split_Info", "Razao_Ganho_C45"],
            ranking_teorico(dados, atributos, TARGET),
        )
    )


def imprimir_raizes_teoricas(dados: list[dict[str, Any]], atributos: list[str]) -> None:
    ranking = ranking_teorico(dados, atributos, TARGET)
    raiz_id3 = max(ranking, key=lambda linha: linha[1])
    raiz_c45 = max(ranking, key=lambda linha: linha[3])

    print("\n=== Raiz teorica escolhida ===")
    print(
        formatar_tabela(
            ["Algoritmo", "Criterio", "Raiz", "Valor"],
            [
                ["ID3", "Maior ganho de informacao", raiz_id3[0], raiz_id3[1]],
                ["C4.5", "Maior razao de ganho", raiz_c45[0], raiz_c45[3]],
            ],
        )
    )


def demonstrar_id_inedito(
    modelo: DecisionTreeClassifier,
    encoders: dict[str, LabelEncoder],
    target_encoder: LabelEncoder,
    atributos: list[str],
    novo_caso: dict[str, Any],
) -> None:
    print("\n=== Teste com ID_Evento inedito ===")
    try:
        classe = predicao_legivel(modelo, encoders, target_encoder, atributos, novo_caso)
        print(f"Predicao inesperadamente aceita: {classe}")
    except ValueError as exc:
        print(
            "O LabelEncoder recusou o ID_Evento=16 porque esse identificador "
            "nao existia no treino."
        )
        print(f"Mensagem tecnica: {exc}")


def main() -> None:
    print("Atividade: Realidade Multidimensional - ID3 vs C4.5")
    print(f"Total de linhas: {len(DADOS)}")
    print(f"Distribuicao do alvo: {dict(Counter(linha[TARGET] for linha in DADOS))}")

    imprimir_dicionario()
    x5, y5, encoders5, target_encoder5 = codificar(DADOS, ATRIBUTOS_5, TARGET)
    imprimir_mapeamento(encoders5, target_encoder5)

    imprimir_ranking(
        DADOS,
        ATRIBUTOS_5,
        "Ranking teorico sem ID_Evento",
    )

    id3 = avaliar_modelo(
        "ID3 simulado no scikit-learn: entropy, sem poda",
        DecisionTreeClassifier(criterion="entropy", random_state=42),
        x5,
        y5,
        ATRIBUTOS_5,
    )

    novo_caso = {
        "Tamanho_PR": "M",
        "Tem_Testes": "S",
        "Complexidade_Ciclo": "B",
        "Autor_Senior": "N",
        "Urgencia": "S",
    }
    classe = predicao_legivel(id3, encoders5, target_encoder5, ATRIBUTOS_5, novo_caso)
    print("\n=== Predicao com inverse_transform ===")
    print(f"Novo caso: {novo_caso}")
    print(f"Saida codificada traduzida: {classe} ({DICIONARIO_DADOS[TARGET]['valores'][classe]})")

    atributos_6 = [*ATRIBUTOS_5, "ID_Evento"]
    x6, y6, encoders6, target_encoder6 = codificar(DADOS_COM_ID, atributos_6, TARGET)

    imprimir_ranking(
        DADOS_COM_ID,
        atributos_6,
        "Experimento do atributo parasita: ID3 vs Razao de Ganho",
    )
    imprimir_raizes_teoricas(DADOS_COM_ID, atributos_6)

    id3_parasita = avaliar_modelo(
        "ID3 com atributo parasita: entropy, sem poda",
        DecisionTreeClassifier(criterion="entropy", random_state=42),
        x6,
        y6,
        atributos_6,
    )

    demonstrar_id_inedito(
        id3_parasita,
        encoders6,
        target_encoder6,
        atributos_6,
        {**novo_caso, "ID_Evento": 16},
    )

    avaliar_modelo(
        "CART podado apenas: entropy, max_depth=5, ccp_alpha=0.01",
        DecisionTreeClassifier(
            criterion="entropy",
            max_depth=5,
            ccp_alpha=0.01,
            random_state=42,
        ),
        x6,
        y6,
        atributos_6,
    )

    atributos_c45 = [atributo for atributo in atributos_6 if atributo != "ID_Evento"]
    x_c45, y_c45, encoders_c45, target_encoder_c45 = codificar(
        DADOS_COM_ID, atributos_c45, TARGET
    )
    c45_didatico = avaliar_modelo(
        "C4.5 didatico: Gain Ratio descarta ID_Evento + poda",
        DecisionTreeClassifier(
            criterion="entropy",
            max_depth=5,
            ccp_alpha=0.01,
            random_state=42,
        ),
        x_c45,
        y_c45,
        atributos_c45,
    )
    classe_c45 = predicao_legivel(
        c45_didatico,
        encoders_c45,
        target_encoder_c45,
        atributos_c45,
        novo_caso,
    )
    print("\n=== Predicao sem o atributo parasita ===")
    print(f"Saida do C4.5 didatico: {classe_c45} ({DICIONARIO_DADOS[TARGET]['valores'][classe_c45]})")

    print("\n=== Reflexao final ===")
    print(
        "O ID3 puro escolhe pelo maior ganho de informacao. Como ID_Evento e "
        "unico, ele cria grupos puros e parece perfeito no historico, mas isso "
        "nao ensina um padrao reutilizavel. A Razao de Ganho pune atributos "
        "com muitos valores e a poda reduz ramos que explicam pouco. Essa e a "
        "diferenca entre decorar registros e aprender uma regra."
    )
    print(
        "Observacao tecnica: o DecisionTreeClassifier do scikit-learn implementa "
        "CART binario e nao possui Gain Ratio nativo. Por isso, profundidade e "
        "ccp_alpha ajudam, mas nao substituem o criterio do C4.5. A tabela "
        "teorica mostra por que o ID_Evento deve ser descartado."
    )


if __name__ == "__main__":
    main()
