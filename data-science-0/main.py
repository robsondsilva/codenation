#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[31]:


#q2()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    linha_coluna = black_friday.shape
    return linha_coluna


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[30]:


def q2():
    qtd_mulheres = black_friday[(black_friday.Age == '26-35') & (black_friday.Gender == 'F')]
    #qtd_unique = qtd_mulheres['User_ID'].nunique()
    return len(qtd_mulheres)


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    qtd_user_unique = black_friday.User_ID.nunique()
    return qtd_user_unique


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    qtd_tipo = black_friday.dtypes.nunique()
    return qtd_tipo


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[26]:


def q5():
    lines_with_null = black_friday.isna().sum().sort_values(ascending=False)[0]/len(black_friday)
    return float(lines_with_null)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    return black_friday.isna().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    return black_friday['Product_Category_3'].value_counts().idxmax()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[28]:


def q8():
    xmin = black_friday['Purchase'].min()
    xmax = black_friday['Purchase'].max()
    black_friday['Purchase_norm'] = (black_friday['Purchase'] - xmin) / (xmax - xmin)
    return float(black_friday['Purchase_norm'].mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[12]:


def q9():
    mean = black_friday['Purchase'].mean()
    std = black_friday['Purchase'].std()
    black_friday['Purchase_pad'] = (black_friday['Purchase'] - mean) / std
    qtd = len(black_friday[(black_friday['Purchase_pad'] <= 1) & (black_friday['Purchase_pad'] >= -1)])
    return qtd


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[13]:


def q10():
    boolean = False
    if len(black_friday[(black_friday['Product_Category_2'].isna() == True) & (black_friday['Product_Category_3'].isna() == False)]) == 0:
        boolean = True
    else:
        boolean = False
    return boolean

