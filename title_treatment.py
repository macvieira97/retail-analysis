import pandas as pd
import numpy as np

#AUX FUNCTIONS

def normalize_name(name):
    # Convert to lowercase and remove leading/trailing spaces
    return name.lower().strip()


def has_numeric(word):
    '''
    Returns True or False if a word has or doesn't have a numeric digit in a sentence, respectively.
    '''
    if any(letter.isnumeric() for letter in word):
        return True
    else:
        return False    


def has_word(title, criteria_list):
    '''
    Returns True or False if a title has or doesn't have a given word from another list, respectively.
    '''
    if title is str:
        title_fragmented = title.split()
        if any(word in title_fragmented for word in criteria_list):
            return True
        else:
            return False


def title_first_numeric(title):
    '''
    Returns the index of the first word in a sentence that has a numeric digit.
    '''
    title_fragmented = title.split()
    for word in title_fragmented:
        if has_numeric(word) == True:
            break
    return title_fragmented.index(word)


def title_first_apparition(title, other_list):
    '''
    Returns the index of the first word in a sentence that also appears in another list.
    '''
    title_fragmented = title.split()
    index = []
    for word in title_fragmented:
        if word in other_list:
            index.append(title_fragmented.index(word))
    return min(index)


def add_ith_name(number:list, df:pd.DataFrame, name_column:str)-> pd.DataFrame:
    '''
    Returns a dataframe containing columns for the ith word of a string.
    '''
    for i in number:
        ith_name = []
        for word in df[name_column]:
            if i-1<len(word.split()):
                ith_name.append(word.split()[i-1])
            else:
                ith_name.append('None')

        df[str(i) + 'names'] = ith_name
    return df

#-------------------------------------------------------
#SPECIFIC FUNCTIONS

def detect_color(cores:list, df:pd.DataFrame, in_column:str, out_column:str)->pd.DataFrame:
    """
    Detects color names in items of a dataframe's column.

    Returns:
        pd.DataFrame: DataFrame containing a new column with the filtered name 
        and a column with the color of the product.
    """
    df['COR'] = np.nan
    nome_sem_cor = []
    for i, nome in enumerate(df[in_column]):
        nome_fragmentado = nome.split()
        cor_completa = str()
        while any((cor:= word) in nome_fragmentado for word in cores):
            cor_completa = cor_completa + ' ' + cor
            nome_fragmentado.remove(cor)
        df.loc[i, 'COR'] = cor_completa
        nome_sem_cor.append(' '.join(nome_fragmentado))
    df[out_column] = nome_sem_cor
    return df


def detect_brand(marcas:list, fab:list, df:pd.DataFrame, in_column:str, out_column:str)->pd.DataFrame:
    """
    Detects brand names in items of a dataframe's column.

    Returns:
        pd.DataFrame: DataFrame containing a new column with the filtered name.
    """
    nome_sem_marca = []
    for i, nome in enumerate(df[in_column]):
        nome_fragmentado = nome.split()
        brand = fab['FABRICANTE'].loc[fab['ID'] == df['ID_FABRICANTE'][i]].to_string()
        while any((brand:= word) in nome_fragmentado for word in marcas):
            nome_fragmentado.remove(brand)          
        nome_sem_marca.append(' '.join(nome_fragmentado))
    df[out_column] = nome_sem_marca
    return df


def remove_prepositions(preposicoes:list, df:pd.DataFrame, in_column:str, out_column:str)->pd.DataFrame:
    """
    Detects prepositions in items of a dataframe's column.

    Returns:
        pd.DataFrame: DataFrame containing a new column with the filtered name.
    """
    nome_sem_preposicao = []
    for nome in df[in_column]:
        nome_fragmentado = nome.split()
        while any((char:= word) in nome_fragmentado for word in preposicoes):
            nome_fragmentado.remove(char)          
        nome_sem_preposicao.append(' '.join(nome_fragmentado))
    df[out_column] = nome_sem_preposicao
    return df

def remove_hanging_prepositions(preposicoes:list, df:pd.DataFrame, in_column:str, out_column:str)->pd.DataFrame:
    """
    Detects prepositions at the end of items title's of a dataframe's column.

    Returns:
        pd.DataFrame: DataFrame containing a new column with the filtered name.
    """
    no_hanging_preposition = []
    for i, nome in enumerate(df[in_column]):
        nome_fragmentado = nome.split()
        if not len(nome_fragmentado) == 0:
            if nome_fragmentado[-1] in preposicoes:
                nome_fragmentado = nome_fragmentado[:-2] 
        no_hanging_preposition.append(' '.join(nome_fragmentado))
    df[out_column] = no_hanging_preposition
    return df

def remove_abbreviations(abbreviations:dict, df:pd.DataFrame, in_column:str, out_column:str)->pd.DataFrame:
    """
    Detects prepositions in items of a dataframe's column.

    Returns:
        pd.DataFrame: DataFrame containing a new column with the filtered name.
    """
    name_no_abb = []
    for nome in df[in_column]:
        nome_fragmentado = nome.split()
        while any((char:= word) in nome_fragmentado for word in list(abbreviations.keys())):
            index = nome_fragmentado.index(char)
            nome_fragmentado[index] = abbreviations[char]
        name_no_abb.append(' '.join(nome_fragmentado))
    df[out_column] = name_no_abb
    return df


def detect_model(df, models, in_column, out_column1, out_column2):
    """
    Splits a string where it is first detected either a word in the criteria list 
    or the first word containig a numerical character.

    Returns:
        pd.DataFrame: DataFrame with two new columns, each containing one part of the original string.
    """
    nome_sem_numero = []
    modelo = []

    for nome in df[in_column]:
        nome_fragmentado = nome.split()[1:]
        first_word = nome.split()[0]

        if has_numeric(nome) or any((char2:= word) in nome_fragmentado for word in models):
            index1 = len(nome) - 1
            index2 = len(nome) - 1
                
            if has_numeric(nome):
                index1 = title_first_numeric(nome) - 1

            if any(word in nome_fragmentado for word in models):
                index2 = title_first_apparition(nome, models) - 1
            
            if index1 < index2:
                index = index1
            else:
                index = index2

            name = first_word + ' '+ ' '.join(nome_fragmentado[:index])
            nome_sem_numero.append(name)
            modelo.append(' '.join(nome_fragmentado[index:]))

        else:
            nome_sem_numero.append(nome.split()[0]+' '+' '.join(nome_fragmentado))
            modelo.append('None')

    df[out_column1] = nome_sem_numero
    df[out_column2] = modelo
    return df

def filter_name(df:pd.DataFrame, fab, cores:list, abb:dict, marcas:list, models:list, preposicoes:list):
    df_final = pd.DataFrame()

    if cores:
        df_final = detect_color(cores, df, 'NOME', 'NOME_FILTRADO')
    if abb:
        df_final = remove_abbreviations(abb, df, 'NOME_FILTRADO', 'NOME_FILTRADO')
    if marcas:
        df_final = detect_brand(marcas, fab,  df_final, 'NOME_FILTRADO', 'NOME_FILTRADO')
    if preposicoes:
        df_final = remove_prepositions(preposicoes, df_final, 'NOME_FILTRADO', 'NOME_FILTRADO')
    if models:
        df_final = detect_model(df_final, models, 'NOME_FILTRADO', 'NOME_FILTRADO', 'MODELO')
        df_final = remove_hanging_prepositions(preposicoes, df, 'NOME_FILTRADO', 'NOME_FILTRADO')

    return df_final

def get_n_grams(text:str, range:tuple)->dict:
    from sklearn.feature_extraction.text import CountVectorizer
    from collections import Counter
    documents = text
    # Create CountVectorizer with bigrams
    vectorizer = CountVectorizer(ngram_range=range)
    # Fit and transform the documents
    ngram_matrix = vectorizer.fit_transform(documents)
    # Get feature names (bigrams)
    feature_names = vectorizer.get_feature_names_out()
    # Convert to dense matrix
    dense_matrix = ngram_matrix.todense()
    # Sum the counts for each bigram across all documents
    ngram_counts = Counter(dict(zip(feature_names, dense_matrix.sum(axis=0).A1)))
    # Display the most common bigrams
    most_common_ngrams = ngram_counts.most_common()
    #print("Most common ngrams:")
    ngrams = []
    counter = []
    for ngram, count in most_common_ngrams:
        #print(f"{ngram}: {count} times")
        ngrams.append(ngram)
        counter.append(count)
    result = {'ngrams': ngrams, 'count': counter}

    return result

def get_bigrams(text):
    import nltk
    from nltk import word_tokenize
    from nltk.util import ngrams

    words = word_tokenize(text)
    bigrams = list(ngrams(words, 2))
    bigrams = [' '.join(bigrams[i]) for i in range(len(bigrams))]
    
    return bigrams

def detect_category(df:pd.DataFrame, in_column:str, dict_of_categories:dict)->pd.DataFrame:
    category = [None] * len(df[in_column])
    
    equipamentos_unigrams = dict_of_categories['equipamentos']['unigrams']
    equipamentos_bigrams = dict_of_categories['equipamentos']['bigrams']

    material_escritorio_unigrams = dict_of_categories['material_de_escritorio']['unigrams']
    material_escritorio_bigrams = dict_of_categories['material_de_escritorio']['bigrams']

    suprimentos_unigrams = dict_of_categories['suprimentos']['unigrams']
    suprimentos_bigrams = dict_of_categories['suprimentos']['bigrams']

    for index, product in enumerate(df[in_column]):
        
        product_unigrams = product.split()
        product_bigrams = get_bigrams(product)
        
        if any((char:=bigram) in product_bigrams for bigram in equipamentos_bigrams):
            category[index] = 'EQUIPAMENTO'
        elif any((char:=bigram) in product_bigrams for bigram in material_escritorio_bigrams):
            category[index] = 'MATERIAL_DE_ESCRITORIO'
        elif any((char:=bigram) in product_bigrams for bigram in suprimentos_bigrams):
            category[index] = 'SUPRIMENTO'
        else:
            if any((char:=word) in product_unigrams for word in equipamentos_unigrams):
                category[index] = 'EQUIPAMENTO'
            elif any((char:=word) in product_unigrams for word in material_escritorio_unigrams):
                category[index] = 'MATERIAL_DE_ESCRITORIO'
            elif any((char:=word) in product_unigrams for word in suprimentos_unigrams):
                category[index] = 'SUPRIMENTO'
            else:
                category[index] = 'None'
    df['CATEGORIA'] = category
    
    return df

def cluster_items(df:pd.DataFrame, in_column:str, list_of_ks:list, k_clusters:int)->pd.DataFrame:

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from nltk.corpus import stopwords
    import matplotlib.pyplot as plt

    product_titles = df[in_column].to_list()
    portuguese_stop_words = list(stopwords.words('portuguese'))

    # Use TF-IDF vectorization to convert titles into numerical features
    vectorizer = TfidfVectorizer(stop_words=portuguese_stop_words)
    X = vectorizer.fit_transform(product_titles)

    if k_clusters == 0:
        # Apply K-means clustering
        inertias = []
        for k in list_of_ks:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        # Plot the elbow curve
        plt.plot(list_of_ks, inertias, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        
        plt.show()

    else:
        # Apply K-means clustering
        inertias = []
        for k in list_of_ks:
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            kmeans.fit(X)
        # Assign cluster labels to the original titles
        cluster_labels = kmeans.labels_

        # Create a DataFrame to display results
        results_df = pd.DataFrame({'product_titles': product_titles, 'Label': cluster_labels})

        df = pd.concat([df, results_df], axis=1)

        return df
    
