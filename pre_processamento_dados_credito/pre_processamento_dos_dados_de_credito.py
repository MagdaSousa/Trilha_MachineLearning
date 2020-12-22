# -*- coding: utf-8 -*-
"""
Pré  processamento da base de dados de histórico de crédito...
Estes dadosz serãoutilizados para prever se o cliente o cliente é
 um bom pagador ou não. E dar ao banco a resposta se é seguro ou não aprovar o 
empréstimo ao cliente. 
"""

import pandas as pd

base = pd.read_csv('credit_data.csv')
#Características das variáveis:
#CLIENTID-> VARIÁVEL CATEGORICA NOMINAL:pois são únicas para cada indivíduo.....

#INCOME-> VARIÁVEL NÚMERICA CONTÍNUA: Valores do tipo float, e como é um salário, entra
# no tipo continua.....

#AGE-> VARIÁVEL NÚMERICA CONTÍNUA :Pois os valores da base, apesar de representarem uma idade
# estão como float, se fossem inteiros, seriam DISCRETA...

#LOAN -> VARIÁVEL NÚMERICA CONTÍNUA->Pois é um valor float, que representa o valor 
# solicitado no empréstimo;

#DEFAULT (discreta)-> É a classe que define os estados( se vai pagar ou não o empréstimo)
#

#-----------------Estatísticas dos dados-----------------------
print(base.describe())

#-------------Tratamento de valores----------------------------

# Ao observamos alguns valores estatísticos, percebemos algumas inconsistências
# por exemplo, o valor mínimo da base , em relação a variável 'age', está negativo
# isso demonstra que há registros com idades zeradas, que ao fazer o cálculo, gerou valores negativos...

#        i#clientid        income          age          loan    c#default
# min     1.000000  20014.489470   -52.423280      1.377630     0.000000


#----------------Localizando os clientes com valores negativos-----------
#Rastreando dados inconsistestes:

clientes_com_idades_negativas = base.loc[base['age']<0]
print(clientes_com_idades_negativas)
"""Verificou-se a existência de 3 clientes com valores inconsistentes,
pois não existe idade negativa:
        i#clientid        income        age         loan         c#default
15          16         50501.726689 -28.218361  3977.287432          0
21          22         32197.620701 -52.423280  4244.057136          0
26          27         63287.038908 -36.496976  9595.286289          0
"""


"""Formas de tratamento dos dados inconsistentes"""


"""1- Apagar a coluna com os dados inconsistentes:(NÃO RECOMENDADO)
Está parte é mais para treinar como apagar a coluna, mas o ideal é tratar os dados, 
pois apagar a coluna ocasiona perda importante de dados que podem ser relevantes"""

base.drop('age',1, inplace=True)
print(base.describe())
"""
'age' -> nome da coluna;
'1'  -> Quer dizer que deseja apagar a coluna inteira;
'implace=True' -> Quer dizer que deseja apagar a coluna e atribuir a alteração a está mesma base de dados,
                  se fosse False, seria atribuida a outra variável, e base ficaria intacta

        i#clientid        income          loan    c#default
count  2000.000000   2000.000000   2000.000000  2000.000000
mean   1000.500000  45331.600018   4444.369695     0.141500
std     577.494589  14326.327119   3045.410024     0.348624
min       1.000000  20014.489470      1.377630     0.000000
25%     500.750000  32796.459717   1939.708847     0.000000
50%    1000.500000  45789.117313   3974.719419     0.000000
75%    1500.250000  57791.281668   6432.410625     0.000000
max    2000.000000  69995.685578  13766.051239     1.000000
"""




""" 2-Apagar somente os registros com o problema
.index-> apagarei o indice também"""

base.drop(base[base.age < 0].index,inplace=True)



""" 3- FAzer a média das idades na base de dados e substituir os valores
inconsistentes pela média"""
base.mean()#média de todas as variáveis da base
base['age'].mean()# média somente da coluna selecionada

# Nesta etapa estamos suponto novamente que a base estpa intacta sem o taratamento
# dos dados negativos, portanto temos que fazer a média somente dos dados maiores do que 0
base['age'][base.age >0].mean()
"""
base['age'][base.age >0].mean()
Out[25]: 40.92770044906149
"""
# Após a descoberta da média basta substituir os valores inconsistentes por ela

base.loc[base.age <0,'age'] =40.92




"""-------------------Tratamento de dados Faltantes-----------------"""

"""Quando as pessoas não passam as informações"""
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

""" Valores faltantes:
    28          29  59417.805406  NaN  2082.625938          0
    30          31  48528.852796  NaN  6155.784670          0
    31          32  23526.302555  NaN  2862.010139          0
"""



"""Separando a base de dados, em previsores e classes"""


previsores = base.iloc[:,1:4].values
# pegar todas as linhas das variáveis 1-income, 2-age, 3- loan, pois a 4 ele não pega
# vamos deixar de lado a variável i#clientid, pois ela é  apenas um identificador, único, que não trará muito valor
# para o algoritimo de machine learning


classes =base.iloc[:,4].values
#pegar todos os valores da coluna defaut que representa as classes dos dados



from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan,strategy='mean') # CTRL +I, para obter a descrição dos dados recebidos pela classe ou métodos
# esta classe fará o preenchimento altomático dos dados faltantes, de acordo com os parâmetros que vc passar

imputer = imputer.fit(previsores[:,0:3])# o fit vai encaixar um valor médio rspequitivo a cada coluna , nos registros faltantes
previsores[:,0:3]= imputer.transform(previsores[:,0:3])# A variável vai receber a transformação que será feita nela mesma


"""---------------------Escalonamento de atributos--------------------------"""
""" Neste caso da base de crédito a diferença entre os valores é muito grande
e em certos algoritmos, isso pode ocasionar certos, problemas, pois  vão acabar levando
em consideração aquelas variáveis como a difrença com o maior voluma.Então é necessário
fazer o escalonamento(aplicar uma fórmula), que diga que os dados , as variáveis estão
na mesma escala...."""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)







