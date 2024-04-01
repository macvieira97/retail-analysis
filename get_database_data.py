import fdb
import pandas as pd
import numpy as np
from datetime import datetime
import os

def get_db_info(query, columns, table):
    """
    Retrieve data from the database.

    Args:
        query (str): SQL query to retrieve data.
        columns (list): List of column names to select.

    Returns:
        pd.DataFrame: DataFrame containing the retrieved data.
    """

    # Dataframe Connections
    conn = fdb.connect(
        host='localhost',
        database=rf'{os.getcwd()}\data\BM.FDB', 
        user='sysdba',
        password='masterkey',
        charset='ISO8859_1'
    )
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()

    # Decode the column values
    decoded_rows = []
    for row in rows:
        decoded_row = []
        for value in row:
            if isinstance(value, bytes):
                try:
                    decoded_value = value.decode('ISO8859_1')
                except UnicodeDecodeError:
                    decoded_value = value.decode('ISO8859_1', errors='replace')
                decoded_row.append(decoded_value)
            else:
                decoded_row.append(value)
        decoded_rows.append(decoded_row)

    # Convert data to a DataFrame
    df = pd.DataFrame(decoded_rows, columns=columns)
    df.to_csv(f'data\silver_tier\{table}.csv')
    cursor.close()
    conn.close()
    return df

def get_pessoa():
    """
    Retrieve information about people from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about people.
    """
    pessoa_columns = ['ID', 'NOME', 'CLIENTE', 'PAIS', 'UF', 'CIDADE', 'BAIRRO', 'RUA', 'CEP']
    PESSOA = f"""SELECT {', '.join(pessoa_columns)} FROM PESSOA;"""
    pessoa_df = get_db_info(PESSOA, pessoa_columns, 'PESSOA')
    return pessoa_df

def get_vendas():
    """
    Retrieve information about sales from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about sales.
    """
    vendas_columns = ['DATA', 'TOTAL', 'ID_CLIENTE']
    VENDAS = f"""SELECT {', '.join(vendas_columns)} FROM SAIDA WHERE ELIMINADO = 'N';"""
    vendas_df = get_db_info(VENDAS, vendas_columns, 'VENDAS')
    vendas_df['TOTAL'] = vendas_df['TOTAL'].astype(float)
    vendas_df['DATA'] = pd.to_datetime(vendas_df['DATA'])
    vendas_df['year-month'] = vendas_df['DATA'].dt.strftime("%Y-%m")
    vendas_df['year'] = vendas_df['DATA'].dt.year
    vendas_df['month'] = vendas_df['DATA'].dt.month
    vendas_df['tipo_de_cliente'] = 'cliente'
    vendas_df.loc[vendas_df['ID_CLIENTE']==0, 'tipo_de_cliente'] = 'balcao'
    return vendas_df

def get_despesas():
    """
    Retrieve information about expenses from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about expenses.
    """
    despesas_columns = ['TOTAL', 'PAGO', 'DT_VENCIMENTO']
    DESPESAS =  f"""SELECT {', '.join(despesas_columns)} FROM CONTASPAGAR WHERE ELIMINADO = 'N';"""
    despesas_df = get_db_info(DESPESAS, despesas_columns, 'DESPESAS')
    despesas_df['TOTAL'] = despesas_df['TOTAL'].astype(float)
    despesas_df['DT_VENCIMENTO'] = pd.to_datetime(despesas_df['DT_VENCIMENTO'])
    despesas_df['year-month'] = despesas_df['DT_VENCIMENTO'].dt.strftime("%Y-%m")
    despesas_df['year'] = despesas_df['DT_VENCIMENTO'].dt.year
    despesas_df['month'] = despesas_df['DT_VENCIMENTO'].dt.month
    return despesas_df

def get_movimento():
    """
    Retrieve information about movement from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about movement.
    """
    actual_date = datetime(2023, 5, 17)
    movimento_columns = ['DATA', 'HORA', 'TIPO', 'ID_PRODUTO', 'SAIDA', 'ENTRADA', 'SALDO']
    MOVIMENTO =  f"""SELECT {', '.join(movimento_columns)} FROM MOVIMENTO;"""
    movimento_df = get_db_info(MOVIMENTO, movimento_columns, 'MOVIMENTO')
    movimento_df['DATETIME'] = pd.to_datetime(movimento_df['DATA'].astype(str) + ' ' + movimento_df['HORA'])
    movimento_df['ultima_movimentacao'] = movimento_df.groupby('ID_PRODUTO')['DATETIME'].transform(np.max)
    ultimo_saldo_ = movimento_df[
    movimento_df['DATETIME']==movimento_df['ultima_movimentacao']
    ][['ID_PRODUTO', 'SALDO']].rename(columns={'SALDO': 'ultimo_saldo'})
    movimento_df = movimento_df.merge(
        ultimo_saldo_, how='outer', on='ID_PRODUTO')
    movimento_df['dias_ultimo_movimento'] = (actual_date - movimento_df['ultima_movimentacao']).dt.days
    return movimento_df

def get_produto():
    """
    Retrieve information about products from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about products.
    """
    produto_columns = ['ID', 'NOME', 'CUSTO_TOTAL', 'PRECO_VENDA', 'ID_FABRICANTE', 'GANHO_LIQUIDO']
    PRODUTO =  f"""SELECT ID, NOME, CUSTO_TOTAL, PRECO_VENDA, ID_FABRICANTE, (PRECO_VENDA - CUSTO_TOTAL) AS GANHO_LIQUIDO FROM PRODUTO;"""
    produto_df = get_db_info(PRODUTO, produto_columns, 'PRODUTO')
    return produto_df

def get_produto_estoque():
    """
    Retrieve information about product stock from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about product stock.
    """
    produto_estoque_columns = ['ID_PRODUTO', 'LOJA', 'LOJAINICIODIA']
    PRODUTO_ESTOQUE =  f"""SELECT {', '.join(produto_estoque_columns)} FROM PRODUTO_ESTOQUE;"""
    produto_estoque_df = get_db_info(PRODUTO_ESTOQUE, produto_estoque_columns, 'PRODUTO_ESTOQUE')

    return produto_estoque_df


def get_boletos_receber():
    """
    Retrieve information about receivable invoices from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about receivable invoices.
    """
    boletos_receber_columns = ['TOTAL', 'PAGO', 'DT_VENCIMENTO', 'ID_PESSOA']
    boletos_receber =  f"""SELECT {', '.join(boletos_receber_columns)} FROM CONTASRECEBER WHERE ELIMINADO = 'N';"""
    boletos_receber_df = get_db_info(boletos_receber, boletos_receber_columns, 'CONTASRECEBER')
    boletos_receber_df['ABERTO'] = 0
    boletos_receber_df.loc[boletos_receber_df['PAGO'] == 0,'ABERTO'] = 1 
    boletos_receber_df['TOTAL'] = boletos_receber_df['TOTAL'].astype(float)
    boletos_receber_df['DT_VENCIMENTO'] = pd.to_datetime(boletos_receber_df['DT_VENCIMENTO'])
    boletos_receber_df['year-month'] = boletos_receber_df['DT_VENCIMENTO'].dt.strftime("%Y-%m")
    boletos_receber_df['year'] = boletos_receber_df['DT_VENCIMENTO'].dt.year
    boletos_receber_df['month'] = boletos_receber_df['DT_VENCIMENTO'].dt.month
    boletos_receber_df['day'] =  boletos_receber_df['DT_VENCIMENTO'].dt.day
    boletos_receber_df.loc[::, 'tipo_de_cliente'] = 'cliente_boleto'
    return boletos_receber_df

def get_product_receipt_info():
    """
    Retrieve information about products in receipt (DAV) from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about products in receipts.
    """
    receipt_product_columns = ['ID_DAV', 'DATA', 'HORA', 'CODIGO', 'DESCRICAO', 'QUANT', 'PRECO']
    receipt_product =  f"""SELECT {', '.join(receipt_product_columns)} FROM D4 WHERE CANCELADO = 'N';"""
    receipt_product_df = get_db_info(receipt_product, receipt_product_columns, 'D4')
    receipt_product_df['PRECO'] = receipt_product_df['PRECO'].astype(float)
    receipt_product_df['DATETIME'] = pd.to_datetime(receipt_product_df['DATA'].astype(str) + ' ' + receipt_product_df['HORA'])
    receipt_product_df['DATA'] = pd.to_datetime(receipt_product_df['DATA'])
    return receipt_product_df

def get_receipt_dav_info():
    """
    Retrieve information about receipt (DAV) from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about receipts.
    """
    receipt_columns = ['ID', 'TITULODAV', 'DATA', 'HORA', 'ID_CLIENTE', 'NOME', 'SUBTOTAL', 'TOTAL', 'PENDENTE']
    receipt_product =  f"""SELECT {', '.join(receipt_columns)} FROM DAV WHERE ELIMINADO = 'N';"""
    receipt_df = get_db_info(receipt_product, receipt_columns, 'DAV')
    receipt_df['DATETIME'] = pd.to_datetime(receipt_df['DATA'].astype(str) + ' ' + receipt_df['HORA'])
    receipt_df['DATA'] = pd.to_datetime(receipt_df['DATA'])
    return receipt_df

def get_estatistica_receita(vendas, boletos_receber):
    """
    Calculate statistics related to revenue.

    Args:
        vendas (pd.DataFrame): DataFrame containing information about sales.
        boletos_receber (pd.DataFrame): DataFrame containing information about receivable invoices.

    Returns:
        pd.DataFrame: DataFrame containing revenue statistics.
    """
    vendas_estatisticas = vendas.groupby('tipo_de_cliente').agg(venda_total = ('TOTAL', 'sum'),
                                                    ticket_medio = ('TOTAL', 'mean'),
                                                    numero_clientes = ('ID_CLIENTE', 'nunique'),
                                                    numero_vendas = ('TOTAL', 'count')).round(0).reset_index()

    boleto_estatisticas = boletos_receber.groupby('tipo_de_cliente').agg(venda_total = ('TOTAL', 'sum'),
                                    ticket_medio = ('TOTAL', 'mean'),
                                    numero_clientes = ('ID_PESSOA', 'nunique'),
                                    numero_vendas = ('TOTAL', 'count')).round(0).reset_index()

    estatisticas_cliente = pd.concat([vendas_estatisticas, boleto_estatisticas])
    estatisticas_cliente = estatisticas_cliente.set_index('tipo_de_cliente')
    return estatisticas_cliente

def get_top_compradores(vendas, pessoa, agrupado_por='NOME'):
    """
    Retrieve information about the top buyers.

    Args:
        vendas (pd.DataFrame): DataFrame containing information about sales.
        pessoa (pd.DataFrame): DataFrame containing information about individuals.
        agrupado_por (str): Column name to group by (default: 'NOME').

    Returns:
        pd.DataFrame: DataFrame containing information about the top buyers.
    """
    vendas_pessoa = vendas.merge(pessoa, how='left', left_on='ID_CLIENTE', right_on='ID')
    #vendas_pessoa = vendas_pessoa[~vendas_pessoa['NOME'].isin(['   CLIENTE BALCÃO', 'JULIANA COMERCIAL LTDA'])]
    top_compradores = vendas_pessoa.groupby(agrupado_por).agg(vendas_total=('TOTAL', 'sum'),
                                                            numero_vendas=('TOTAL', 'count'),
                                                            ticket_medio=('TOTAL', 'mean'))
    return top_compradores.sort_values(by='vendas_total', ascending=False)


def vendas_info(saida_df):
    """
    Calculate information about sales.

    Args:
        saida_df (pd.DataFrame): DataFrame containing information about sales.

    Returns:
        tuple: A tuple containing three DataFrames:
            - DataFrame with general sales information
            - DataFrame with customer-specific sales information
            - DataFrame with counter sales information
    """
    cliente = saida_df['ID_CLIENTE'] !=0
    balcao = saida_df['ID_CLIENTE'] == 0
    geral_ = saida_df.groupby(['year-month']).agg(vendas_total = ('TOTAL', 'sum'),
                                                        total_de_compras = ('TOTAL', 'count'),
                                                        ticket_medio_geral = ('TOTAL', 'mean'))
    clientes_ = saida_df[cliente].groupby(['year-month']).agg(vendas_total_cliente = ('TOTAL', 'sum'),
                                                              total_de_compras_cliente = ('TOTAL', 'count'),
                                                              clientes_unicos = ('ID_CLIENTE', 'nunique'),
                                                              ticket_medio_cliente = ('TOTAL', 'mean'))

    balcao_ = saida_df[balcao].groupby(['year-month']).agg(clientes_unicos = ('ID_CLIENTE', 'nunique'),
                                                           vendas_total_balcao = ('TOTAL', 'sum'),
                                                           total_de_compras_balcao = ('TOTAL', 'count'),
                                                           ticket_medio_balcao = ('TOTAL', 'mean'))
    return geral_, clientes_, balcao_

# ADD POR MARI
def get_fabricante():
    """
    Retrieve information about products from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about fabricantes.
    """
    fabricante_columns = ['ID', 'FABRICANTE', 'ELIMINADO']
    FABRICANTE =  f"""SELECT ID, FABRICANTE, ELIMINADO FROM FABRICANTE;"""
    fabricante_df = get_db_info(FABRICANTE, fabricante_columns, 'FABRICANTE')
    return fabricante_df

def get_saida():
    """
    Retrieve information about products from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about saidas.
    """
    saida_columns = ['ID', 'DATA', 'HORA', 'ID_CLIENTE', 'TOTALPRODUTOS']
    SAIDA =  f"""SELECT ID,  DATA, HORA, ID_CLIENTE, TOTALPRODUTOS FROM SAIDA;"""
    saida_df = get_db_info(SAIDA, saida_columns, 'SAIDA')
    return saida_df

def get_saida_produto():
    """
    Retrieve information about products from the database.

    Returns:
        pd.DataFrame: DataFrame containing information about saida_produtos.
    """
    saida_produto_columns = ['ID', 'ID_SAIDA', 'ID_PRODUTO', 'QUANT', 'PRECO', 'TOTAL', 'CANCELADO']
    SAIDA_PRODUTO =  f"""SELECT ID,  ID_SAIDA, ID_PRODUTO, QUANT, PRECO, TOTAL, CANCELADO FROM SAIDA_PRODUTO;"""
    saida_produto_df = get_db_info(SAIDA_PRODUTO, saida_produto_columns, 'SAIDA_PRODUTO')
    return saida_produto_df


