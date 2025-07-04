# 1) Import de bibliotecas

# Configuração para não exibir os warnings
import warnings
warnings.filterwarnings("ignore")

# Imports de bibliotecas de manipulação de dados
import pandas as pd  # Manipulação de dados em tabelas
import numpy as np   # Funções matemáticas e arrays

# Imports para visualização
import matplotlib.pyplot as plt  # Criação de gráficos e visualizações
import seaborn as sns

# Imports para pré-processamento
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV  # Particionamento dos dados e validação cruzada
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

# 2) Carga do DataSet por url

# url a importar
url_dados = 'https://raw.githubusercontent.com/geovane186/MVP_Data_Analysis_Obesity_Levels/refs/heads/main/DataSet/ObesityDataSet_raw_and_data_sinthetic_training.csv'

# Carga do dataset através do csv
obesityDataSet = pd.read_csv(url_dados)

# Verifica o tipo de obesityDataSet
print('Classe do DataSet:',type(obesityDataSet), '\n')

# Exibe as 5 primeiras linhas
print(obesityDataSet.head(), '\n')

# 3) Checar total e tipo das instancias do dataset, e valores nulos

# Verificar total e tipo das instancias
print(f"Total de instâncias: {len(obesityDataSet)}")
print("\nTipos de dados por coluna:")
print(obesityDataSet.info())

# Verificar valores faltantes
print("\nChecagem de nulos:")
print(obesityDataSet.isnull().sum())

# Verificar valores únicos por coluna categórica
print("\nChecagem de valores inconsistentes:")
for col in obesityDataSet.select_dtypes(include='object').columns:
    print(f'\nColuna: {col}')
    print(obesityDataSet[col].unique())

# 4) Checar balanceamento das classes

# gráfico de barras simples
plt.figure(figsize=(17, 5))
sns.countplot(x='NObeyesdad', data=obesityDataSet)
plt.title('Distribuição dos niveis de obesidade')
plt.xlabel('Nivel de obesidade')
plt.ylabel('Contagem')
plt.show()

# Frequência absoluta e relativa
class_counts = obesityDataSet['NObeyesdad'].value_counts()
class_percent = obesityDataSet['NObeyesdad'].value_counts(normalize=True) * 100

class_distribution = pd.DataFrame({
    'Contagem': class_counts,
    'Porcentagem (%)': class_percent.round(2)
})

print(class_distribution)

# 5) Estatísticas descritivas básicas do dataset
obesityDataSet.describe()

# 6) Exibição da Média

# Lista dos atributos numéricos
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Calcula a média de cada atributo
means = obesityDataSet[num_cols].mean()

# Gráfico de barras
plt.figure(figsize=(10, 6))
means.plot(kind='bar', color='skyblue')
plt.title('Média dos atributos numéricos')
plt.ylabel('Média')
plt.xticks(rotation=45)
plt.show()

# 7) Exibição do desvio Padrão

# Lista dos atributos numéricos
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Calcula o desvio-padrão de cada atributo
stds = obesityDataSet[num_cols].std()

# Gráfico de barras
plt.figure(figsize=(10, 6))
stds.plot(kind='bar', color='salmon')
plt.title('Desvio-padrão dos atributos numéricos')
plt.ylabel('Desvio-padrão')
plt.xticks(rotation=45)
plt.show()

# 8) Histograma de Idade

plt.figure(figsize=(8, 6))
sns.histplot(obesityDataSet['Age'], kde=True)
plt.title('Distribuição: Age')
plt.xlabel('Age')
plt.ylabel('Frequência')
plt.show()

# 9) Histograma da Altura

plt.figure(figsize=(8, 6))
sns.histplot(obesityDataSet['Height'], kde=True)
plt.title('Distribuição: Height')
plt.xlabel('Height')
plt.ylabel('Frequência')
plt.show()

# 10) Histograma do Peso

plt.figure(figsize=(8, 6))
sns.histplot(obesityDataSet['Weight'], kde=True)
plt.title('Distribuição: Weight')
plt.xlabel('Weight')
plt.ylabel('Frequência')
plt.show()

# 11) Histograma da Frequência de Consumo de Vegetais

plt.figure(figsize=(8, 6))
sns.histplot(obesityDataSet['FCVC'], kde=True)
plt.title('Distribuição: FCVC')
plt.xlabel('FCVC')
plt.ylabel('Frequência')
plt.show()

# 12) Histograma do Número de Refeições Principais Diárias

plt.figure(figsize=(8, 6))
sns.histplot(obesityDataSet['NCP'], kde=True)
plt.title('Distribuição: NCP')
plt.xlabel('NCP')
plt.ylabel('Frequência')
plt.show()

# 13) Histograma do Consumo Diário de Água

plt.figure(figsize=(8, 6))
sns.histplot(obesityDataSet['CH2O'], kde=True)
plt.title('Distribuição: CH2O')
plt.xlabel('CH2O')
plt.ylabel('Frequência')
plt.show()

# 14) Histograma da Frequência de Atividade Física

plt.figure(figsize=(8, 6))
sns.histplot(obesityDataSet['FAF'], kde=True)
plt.title('Distribuição: FAF')
plt.xlabel('FAF')
plt.ylabel('Frequência')
plt.show()

# 15) Histograma do Tempo de Uso de Dispositivos Tecnológicos

plt.figure(figsize=(8, 6))
sns.histplot(obesityDataSet['TUE'], kde=True)
plt.title('Distribuição: TUE')
plt.xlabel('TUE')
plt.ylabel('Frequência')
plt.show()

# 16) Estatísticas descritivas agrupadas por nivel de obesidade

# Define a ordem correta
ordered_classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

# Define a variável como categórica com ordem específica
obesityDataSet['NObeyesdad'] = pd.Categorical(
    obesityDataSet['NObeyesdad'],
    categories=ordered_classes,
    ordered=True
)

num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

for col in num_cols:
    print(f"\n=== {col} ===")
    print(
        obesityDataSet.groupby('NObeyesdad', observed=True)[col]
        .describe()[['mean', '50%', 'std', 'min', 'max']]
        .rename(columns={'50%': 'median'})
    )

# 17) Boxplot da Idade (Age) por nível de obesidade

ordered_classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

plt.figure(figsize=(15, 6))
sns.boxplot(
    x='NObeyesdad',
    y='Age',
    data=obesityDataSet,
    order=ordered_classes
)
plt.title('Idade por Nível de Obesidade')
plt.xlabel('Nível de Obesidade')
plt.ylabel('Idade')
plt.show()

# 18) Boxplot de Altura (Height) por nível de obesidade

ordered_classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

plt.figure(figsize=(15, 6))
sns.boxplot(
    x='NObeyesdad',
    y='Height',
    data=obesityDataSet,
    order=ordered_classes
)
plt.title('Altura por Nível de Obesidade')
plt.xlabel('Nível de Obesidade')
plt.ylabel('Altura')
plt.show()

# 19) Boxplot de Peso (Weight) por nível de obesidade

ordered_classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

plt.figure(figsize=(15, 6))
sns.boxplot(
    x='NObeyesdad',
    y='Weight',
    data=obesityDataSet,
    order=ordered_classes
)
plt.title('Peso por Nível de Obesidade')
plt.xlabel('Nível de Obesidade')
plt.ylabel('Peso')
plt.show()

# 20) Boxplot da Frequência de Consumo de Vegetais (FCVC) por nível de obesidade

ordered_classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

plt.figure(figsize=(15, 6))
sns.boxplot(
    x='NObeyesdad',
    y='FCVC',
    data=obesityDataSet,
    order=ordered_classes
)
plt.title('Frequência de Consumo de Vegetais por Nível de Obesidade')
plt.xlabel('Nível de Obesidade')
plt.ylabel('Frequência de Consumo de Vegetais')
plt.show()

# 21) Boxplot do Número de Refeições Principais (NCP) por nível de obesidade

ordered_classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

plt.figure(figsize=(15, 6))
sns.boxplot(
    x='NObeyesdad',
    y='NCP',
    data=obesityDataSet,
    order=ordered_classes
)
plt.title('Número de Refeições Principais por Nível de Obesidade')
plt.xlabel('Nível de Obesidade')
plt.ylabel('Número de Refeições Principais')
plt.show()

# 22) Boxplot do Consumo de Água (CH2O) por nível de obesidade

ordered_classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

plt.figure(figsize=(15, 6))
sns.boxplot(
    x='NObeyesdad',
    y='CH2O',
    data=obesityDataSet,
    order=ordered_classes
)
plt.title('Consumo de Água por Nível de Obesidade')
plt.xlabel('Nível de Obesidade')
plt.ylabel('Consumo de Água')
plt.show()

# 23) Boxplot da Frequência de Atividade Física (FAF) por nível de obesidade

ordered_classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

plt.figure(figsize=(15, 6))
sns.boxplot(
    x='NObeyesdad',
    y='FAF',
    data=obesityDataSet,
    order=ordered_classes
)
plt.title('Frequência de Atividade Física por Nível de Obesidade')
plt.xlabel('Nível de Obesidade')
plt.ylabel('Frequência de Atividade Física')
plt.show()

# 24) Boxplot da Tempo de Uso de Dispositivos Tecnológicos (TUE) por nível de obesidade

ordered_classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

plt.figure(figsize=(15, 6))
sns.boxplot(
    x='NObeyesdad',
    y='TUE',
    data=obesityDataSet,
    order=ordered_classes
)
plt.title('Tempo de Uso de Dispositivos Tecnológicos por Nível de Obesidade')
plt.xlabel('Nível de Obesidade')
plt.ylabel('Tempo de Uso de Dispositivos Tecnológicos')
plt.show()

# 25) Exibição da Matriz de correlação

# Lista de colunas numéricas
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Calcula a matriz de correlação entre essas colunas
corr_matrix = obesityDataSet[num_cols].corr()

# Exibe a tabela no console
print("\nMatriz de Correlação entre atributos numéricos:")
print(corr_matrix)

# 26) Exibição do Heatmap da Matriz de correlação

# Lista de colunas numéricas
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Calcula a matriz de correlação
corr_matrix = obesityDataSet[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    square=True,
    linewidths=.5
)
plt.title('Matriz de Correlação entre Atributos Numéricos')
plt.show()

# 27) Checar e remover duplicatas

# Verificar quantas linhas duplicadas existem
num_duplicates = obesityDataSet.duplicated().sum()
print(f"Número de linhas duplicadas encontradas: {num_duplicates}")

# Remover duplicatas, mantendo apenas a primeira ocorrência
obesityDataSet_cleaned = obesityDataSet.drop_duplicates()

# Conferir o novo shape
print(f"Shape do dataset após remoção: {obesityDataSet_cleaned.shape}")

# 28) Divisão inicial do dataset limpo (Holdout e Validação Cruzada)

seed = 42 # Semente para reprodutibilidade

testSize = 0.20 # tamanho do conjunto de teste

# Separação em conjuntos de treino e teste
X = obesityDataSet_cleaned.drop(columns=['NObeyesdad'])
y = obesityDataSet_cleaned['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=testSize, shuffle=True, random_state=seed, stratify=y) # holdout com estratificação

# Parâmetros e partições da validação cruzada
num_particoes = 10
kfold = StratifiedKFold(n_splits=num_particoes, shuffle=True, random_state=seed) # validação cruzada com estratificação

# 29) Codificação de atributos

np.random.seed(42)

# Codificação personalizada (usando funções auxiliares)
def encode_ordinal(data, columns, mapping_dicts):
    for col, mapping in zip(columns, mapping_dicts):
        data[col] = data[col].map(mapping)
    return data

# Mapeamento explícito para variáveis categóricas
ordinal_mappings = [
    {'Female': 0, 'Male': 1}, # Gender
    {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},  # CAEC
    {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}  # CALC
]

# Definir colunas categóricas
ordinal_cols = ['Gender', 'CAEC', 'CALC']
nominal_cols = ['MTRANS']
categorical_simple_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']

# Criar transformações
ordinal_transformer = FunctionTransformer(
    encode_ordinal, kw_args={'columns': ordinal_cols, 'mapping_dicts': ordinal_mappings}
)

nominal_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# Para as variáveis categóricas simples (LabelEncoder em cada coluna)
simple_transformer = FunctionTransformer(lambda df: df.apply(lambda col: LabelEncoder().fit_transform(col)))

preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_transformer, ordinal_cols),
        ('nominal', nominal_transformer, nominal_cols),
        ('simple', simple_transformer, categorical_simple_cols)
    ],
    remainder='passthrough' # Manter outras colunas inalteradas
)

# Fit no treino
preprocessor.fit(X_train)

# Transformar treino e teste
X_train_encoded = preprocessor.transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Codificação da variável target
target_mapping = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}

# Codificar a variável target
y_train_encoded = y_train.map(target_mapping)
y_test_encoded = y_test.map(target_mapping)

# Reconstruir nomes das colunas transformadas
# Nomes do OneHotEncoder
onehot_encoder = preprocessor.named_transformers_['nominal']
onehot_feature_names = onehot_encoder.get_feature_names_out(nominal_cols)

# Ordinais e Simples mantêm nomes
ordinal_feature_names = ordinal_cols
simple_feature_names = categorical_simple_cols

# Pegar as colunas que passaram direto
passthrough_cols = [col for col in X_train.columns if col not in ordinal_cols + nominal_cols + categorical_simple_cols]

# Juntar tudo na ordem exata do preprocessor
final_feature_names = list(ordinal_feature_names) + list(onehot_feature_names) + list(simple_feature_names) + passthrough_cols

print("\nNomes finais das features transformadas:")
print(final_feature_names)

# Converter X_train_encoded para DataFrame
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=final_feature_names)

# 30) Normalização dos atributos

# Inicializar o MinMaxScaler
scaler_norm = MinMaxScaler()

numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Aprende min e max APENAS de X_train
scaler_norm.fit(X_train[numeric_cols])
X_train_normalized = scaler_norm.transform(X_train[numeric_cols])
# Usa a média e o desvio padrão aprendidos de X_train
X_test_normalized = scaler_norm.transform(X_test[numeric_cols])

# Exibir as primeiras linhas dos dados normalizados (como DataFrame para melhor visualização)
df_normalized = pd.DataFrame(X_train_normalized, columns=numeric_cols)

print("\nPrimeiras 5 linhas dos dados normalizados (treino):")
print(df_normalized.head())

# Histograma para cada atributo numérico normalizado
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_normalized[col], kde=True, bins=20)
    plt.title(f'Distribuição de {col} (Normalizado)')
    plt.xlabel(f'{col} (Normalizado)')
    plt.ylabel('Frequência')
    plt.show()

# 31) Padronização dos atributos

# Inicializar o StandardScaler
scaler_std = StandardScaler()

numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Aprende média e desvio padrão APENAS de X_train
scaler_std.fit(X_train[numeric_cols])

# Padronizar os dados (somente colunas numéricas)
X_train_standardized = scaler_std.transform(X_train[numeric_cols])
X_test_standardized = scaler_std.transform(X_test[numeric_cols])

# Criar DataFrame para visualização
df_standardized = pd.DataFrame(X_train_standardized, columns=numeric_cols)

print("\nPrimeiras 5 linhas dos dados padronizados (treino):")
print(df_standardized.head())

# Visualização da distribuição após a padronização para os atributos numéricos
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_standardized[col], kde=True, bins=20)
    plt.title(f'Distribuição de {col} (Padronizado)')
    plt.xlabel(f'{col} (Padronizado)')
    plt.ylabel('Frequência')
    plt.show()

# 32) SelectKBest (com ANOVA F-Value)

# Define numero de atributos
k = 10

# Seleção de atributos com SelectKBest
selector_kbest = SelectKBest(score_func=f_classif, k=k)
X_train_kbest = selector_kbest.fit_transform(X_train_encoded_df, y_train_encoded)

selected_mask = selector_kbest.get_support()
selected_features = np.array(X_train_encoded_df.columns)[selected_mask]

print("\nAtributos selecionados pelo SelectKBest:")
print(selected_features)

print("\nScores (F-value) de cada atributo:")
print(selector_kbest.scores_)

# 33) Coeficientes da Regressão Logística (Eliminação Recursiva de Atributos)

# Inicializa o modelo de regressão logística
logreg = LogisticRegression(max_iter=200, solver='liblinear', random_state=42)

# Inicializa o RFE para selecionar 10 atributos
rfe = RFE(estimator=logreg, n_features_to_select=10)

# Ajusta o RFE nos dados de treino
rfe.fit(X_train_encoded, y_train_encoded)

# Máscara de seleção dos atributos
selected_rfe_mask = rfe.support_

# Índices dos atributos selecionados
selected_rfe_indices = rfe.get_support(indices=True)

# Para ver os nomes dos atributos selecionados:
selected_rfe_features = np.array(X_train_encoded_df.columns)[selected_rfe_mask]

print("\nAtributos selecionados pelo RFE:")
print(selected_rfe_features)

print("\nRanking de cada atributo (1 = selecionado, maior = menos importante):")
print(rfe.ranking_)

# 34) SelectFromModel com ExtraTreesClassifier (Importância de Atributos com ExtraTrees)

# Cria o modelo ExtraTrees
extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)

# Treina o modelo nos dados codificados
extra_trees.fit(X_train_encoded, y_train_encoded)

# Usa o SelectFromModel para selecionar os atributos com maior importância
selector_sfm = SelectFromModel(extra_trees, prefit=True)

# Aplica o SelectFromModel ao treino
X_train_sfm = selector_sfm.transform(X_train_encoded)

# Ver atributos selecionados
selected_mask_sfm = selector_sfm.get_support()
selected_features_sfm = X_train_encoded_df.columns[selected_mask_sfm]

print("\nAtributos selecionados pelo SelectFromModel (ExtraTreesClassifier):")
print(selected_features_sfm)

# Visualizar importâncias de todos os atributos
importances = extra_trees.feature_importances_
for name, score in zip(X_train_encoded_df.columns, importances):
    print(f"{name}: {score:.4f}")