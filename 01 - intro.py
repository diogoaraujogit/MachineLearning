'''
Introduction to Statical Learning - Gareth James
Método de análise de dados que automatiza o processo de criação de modelos
Usando algoritmos que iterativamente aprendem dos dados, Machine Learning permite que computadores
encontrem padrões escondidos nos dados sem terem sido programados para isso

Existem 3 tipos de algoritmos de Machine Learning:
- Supervised Learning: você tem parâmetros rotulados que são usados para construir o modelo e tentar predizer os demais
rótulos, baseados nos parâmetros apenas.
    Exemplo: você tem características técnicas de peças de equipamentos que falharam "F" e não falharam "NF", e quer
    predizer o comportamento das demais peças
O algoritmo de aprendizado recebe entradas com as saídas corretas e ajusta o seu modelo de forma iterativa para que o mesmo
se adapte as condições apresentadas no conjunto de dados de treino
Então, o algoritmo irá conferir a precisão do modelo criado usando o conjunto de dados de teste


- Unsupervised Learning: você possui apenas os parâmetros, sem rótulos, e quer encontrar subgrupos dentro dos dados que
possuam algum tipo de semelhança
Usado quando não se tem classificações prévias. A resposta correta não é dita ao algoritmo, cabendo a ele encontrar
padrões nos dados e agrupá-los/classificá-los baseados similaridades no conjunto de parâmetros
Técnicas populares incluem mapas auto-organizáveis: k-means clustering e singular value decomposition
    Exemplos: Estes algoritmos também são usados para segmentar texto em tópicos, identificação de outliers em conjuntos
    de dados e recomendação de itens à clientes

- Reinforcement Learning: algoritmos que aprendem a executar ações baseados em experiências do mesmo com algum meio
    Exemplo: Este tipo de algoritmo é usado principalmente em robótica, jogos e navegação
Através deste método, o algoritmo aprende através da tentativa e erro quais pares de estado-ação obtém a maior recompensa
no longo prazo
O objetivo do agente é escolher ações que maximizem a recompensa esperada dada uma determinada quantidade de tempo
O agente irá, desta forma, criar  uma política de tomada de decisões, baseadas no seu atual estado
'''