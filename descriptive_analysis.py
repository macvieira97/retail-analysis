import pandas as pd

def show_top_sales(produtos_mais_vendidos:pd.DataFrame, time_frame:str, top_n:int):
    import plotly.subplots as subplot
    fig = subplot.make_subplots(rows=1, cols=2, horizontal_spacing=0.1)
    produtos_mais_vendidos = produtos_mais_vendidos.sort_values(by='QUANT_TOTAL', ascending=False).reset_index(drop=True)
    x_names_quant = produtos_mais_vendidos['GRUPO_PRODUTO'].to_list()[:top_n]
    quantity_bar = fig.add_bar(
        x=x_names_quant,
        y=produtos_mais_vendidos['QUANT_TOTAL'],
        marker=dict(color= 'rgba(100, 159, 237, 1)'),
        showlegend=False,
        row=1, col=1)
    quantity_bar.add_annotation(
        text='<span style="color:rgba(100, 159, 237, 1);"># OF ITEMS</span> PER PRODUCT TYPE',
        xref='paper', yref='paper',
        x=-0.02, y=1.15,  # Adjust the x and y values to position the subtitle
        showarrow=False,
        font=dict(size=12, color='black'))
    quantity_bar.layout.update(title=dict(text="TOP " + str(top_n) + " SELLING PRODUCTS IN THE LAST " + time_frame), title_font=dict(size=18),
                    plot_bgcolor='rgba(0, 0, 0, 0)',                  
                    width = 600, height=500)
    produtos_mais_vendidos = produtos_mais_vendidos.sort_values(by='FATURAMENTO_TOTAL', ascending=False).reset_index(drop=True)
    x_names_fat = produtos_mais_vendidos['GRUPO_PRODUTO'].to_list()[:top_n]
    revenue_bar = fig.add_bar(
        x=x_names_fat,
        y=produtos_mais_vendidos['FATURAMENTO_TOTAL'],
        marker=dict(color= 'rgb(114, 172, 77)'),
        showlegend=False,
        row=1, col=2)
    revenue_bar.add_annotation(
        text='<span style="color:rgb(114, 172, 77);">REVENUE</span> PER PRODUCT TYPE',
        xref='paper', yref='paper',
        x=0.7, y=1.15,  # Adjust the x and y values to position the subtitle
        showarrow=False,
        font=dict(size=12, color='black'))
    revenue_bar.layout.update(title=dict(text="TOP " + str(top_n) + " SELLING PRODUCTS IN THE LAST " + time_frame), title_font=dict(size=18),
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    width = 600, height=500)
    fig.layout.update({'width' : 1000, 'height': 500})
    fig.show()

from datetime import datetime

def top_n_sales(df, df_vendas:pd.DataFrame, months_ago:int, last_sell:datetime):
    from dateutil.relativedelta import relativedelta
    # DROP SALES FROM BEFORE TIMEFRAME
    df_timeframe = df_vendas.copy()
    df_timeframe.drop(df_timeframe.index[df_timeframe['DATA'] < (last_sell - relativedelta(months=months_ago))], inplace=True)

    # ADICIONANDO À TABELA A QUANTIDADE TOTAL DE PRODUTOS VENDIDOS DO MESMO TIPO
    sum_qntd_product = df_timeframe.groupby('GRUPO_PRODUTO').agg(sum=('QUANT','sum')).reset_index()
    df_timeframe = pd.merge(df_timeframe, sum_qntd_product, on='GRUPO_PRODUTO', how='outer')
    df_timeframe = df_timeframe.sort_values(by='sum', ascending=False).reset_index(drop=True)
    df_timeframe.rename(columns={'sum': 'QUANT_TOTAL'}, inplace=True)

    # ADICIONANDO À TABELA O FATURAMENTO TOTAL DE PRODUTOS VENDIDOS DO MESMO TIPO
    preco_total = df_timeframe.groupby('GRUPO_PRODUTO').agg(sum=('TOTAL','sum')).reset_index()
    df_timeframe = pd.merge(df_timeframe, preco_total, on='GRUPO_PRODUTO', how='outer')
    df_timeframe = df_timeframe.sort_values(by='sum', ascending=False).reset_index(drop=True)
    df_timeframe.rename(columns={'sum': 'FATURAMENTO_TOTAL'}, inplace=True)

    produtos_mais_vendidos = df_timeframe[['GRUPO_PRODUTO', 'CATEGORIA', 'QUANT_TOTAL', 'FATURAMENTO_TOTAL']].reset_index(drop=True).copy()
    produtos_mais_vendidos = produtos_mais_vendidos.dropna().reset_index(drop=True)
    produtos_mais_vendidos = produtos_mais_vendidos.drop_duplicates().reset_index(drop=True)
    return produtos_mais_vendidos, df_timeframe

def define_time_periods(df, last_sell, intervals, step):
    '''
    Returns dataframe with an additional column indicating since when each data point was effected.
    The indicated time periods are separated according to an input variable, so it can be modified according to especific requirements.
    '''
    from dateutil.relativedelta import relativedelta
    if step == 'day':
        date_intervals = [last_sell - relativedelta(days=x) for x in intervals]
    elif step == 'week':
        date_intervals = [last_sell - relativedelta(weeks=x) for x in intervals]
    elif step == 'month':
        date_intervals = [last_sell - relativedelta(months=x) for x in intervals]

    # CATEGORIZE EACH SELLING WITHIN THEIR RESPECTIVE TIME PERIOD
    df['INTERVAL'] = str(last_sell)
    indexes = []
    for i in range(len(date_intervals)):
        indexes = list(df.loc[(df['DATA'] >= date_intervals[i]) ].index)
        df.loc[indexes, 'INTERVAL'] = str(date_intervals[i])
    return df


def product_over_type(df_vendas, product):
    # ADDS A COLUMN WITH THE TOTAL QUANTITY OF EACH PARTICULAR PRODUCT
    df = df_vendas[df_vendas['GRUPO_PRODUTO']==product].copy()
    quant_total = df.groupby('NOME').agg(sum=('QUANT','sum')).reset_index()
    df = pd.merge(df, quant_total, on='NOME', how='outer')
    df = df.sort_values(by='sum', ascending=False).reset_index(drop=True)
    df.rename(columns={'sum': 'QUANTIDADE_TOTAL'}, inplace=True)

    # ADDING A COLUMN WITH THE TOTAL REVENUE PER PARTICULAR PRODUCT
    revenue_total = df.groupby('NOME').agg(sum=('TOTAL','sum')).reset_index()
    df = pd.merge(df, revenue_total, on='NOME', how='outer')
    df.rename(columns={'sum': 'REVENUE_TOTAL'}, inplace=True)

    # SAVING IT IN A NEW DATAFRAME CONTAINING ONLY THE SAME TYPE OF PRODUCT 
    df_product = df[['NOME', 'REVENUE_TOTAL', 'QUANTIDADE_TOTAL']].copy()
    df_product = df_product.dropna().reset_index(drop=True)
    df_product = df_product.drop_duplicates().reset_index(drop=True)
    highest_sell = df_product['NOME'][0]
    # TESTING FOR COHERENCE
    #total_polaseal = df.groupby('TOTAL_POLASEAL').agg(min=('TOTAL_POLASEAL', 'min')).agg(sum=('min', sum))
    #total_polaseal
    return df_product


def plot_product_over_type_dist(df_vendas, product):
    import plotly.graph_objects as go
    # FILTERS PRODUCT IN PURCHASE DATAFRAME
    df = df_vendas[df_vendas['GRUPO_PRODUTO']==product].copy()
    # COMPUTES TOTAL SUM OF EACH TYPE OF PRODUCT IN THE GENERAL PRODUCT CLASSIFICATION
    quant_total = df.groupby('NOME').agg(sum=('QUANT','sum')).reset_index()
    df = pd.merge(df, quant_total, on='NOME', how='outer')
    df = df.sort_values(by='sum', ascending=False).reset_index(drop=True)
    df.rename(columns={'sum': 'QUANTIDADE_TOTAL'}, inplace=True)
    dist_product = df[['NOME', 'QUANTIDADE_TOTAL']].reset_index(drop=True).copy()
    dist_product = dist_product.dropna().reset_index(drop=True)
    dist_product = dist_product.drop_duplicates().reset_index(drop=True)
    bar = go.Figure(go.Bar(
        x= list(range(len(dist_product['QUANTIDADE_TOTAL']))),
        y=dist_product['QUANTIDADE_TOTAL'],
        marker=dict(color= 'rgb(114, 172, 77)')))

    bar.layout.update(title=dict(text=str(product.upper()) + ' SALES DISTRIBUTION'), title_font=dict(size=18),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                xaxis=dict(title=str(product.upper()) + ' TYPE', title_standoff=10),
                yaxis=dict(title='UNITS SOLD'),
                width = 600, height=500)
    bar.show()


def add_missing_dates(df_product, product, last_sell, intervals, step):
    from dateutil.relativedelta import relativedelta
    if step == 'day':
        date_intervals = [last_sell - relativedelta(days=x) for x in intervals]
    elif step == 'week':
        date_intervals = [last_sell - relativedelta(weeks=x) for x in intervals]
    elif step == 'month':
        date_intervals = [last_sell - relativedelta(months=x) for x in intervals]
    # CATEGORIZE EACH SELLING WITHIN THEIR RESPECTIVE TIME PERIOD
    df_adjusted = pd.DataFrame()
    quantidades = [0]*len(date_intervals)
    df_adjusted['DATA'] = [last_sell - relativedelta(days=x) for x in intervals]
    df_adjusted['GRUPO_PRODUTO'] = product
    df_adjusted['QUANT_PER_TIME_STEP'] = quantidades
    df_adjusted['INTERVAL'] = date_intervals

    for i in range(len(date_intervals)):
        if date_intervals[i] in set(df_product['DATA']):
            index_df_product = df_product.loc[(df_product['DATA'] == date_intervals[i])].index
            df_adjusted.loc[i, 'QUANT_PER_TIME_STEP'] = int(df_product.loc[index_df_product,'QUANT_PER_TIME_STEP'].iloc[0])
    return df_adjusted

def to_date(x):
    import datetime
    return datetime.datetime.strptime(x, "%Y-%m-%d").date()


def product_over_time(product, df_vendas, last_sell, intervals, step):
    # FILTERS PRODUCT IN PURCHASE DATAFRAME
    df = df_vendas[df_vendas['GRUPO_PRODUTO']==product].copy()
    # DEFINING THE TIMEFRAME
    df = define_time_periods(df, last_sell, intervals, step)
    # COMPUTES TOTAL PURCHASES FOR EACH TIMEFRAME STEP
    sales_per_time_step = df.groupby('INTERVAL').agg(sum=('QUANT','sum')).reset_index()
    df = pd.merge(df, sales_per_time_step, on='INTERVAL', how='outer')
    df.rename(columns={'sum': 'QUANT_PER_TIME_STEP'}, inplace=True) 
    # SAVES IN A NEW DATAFRAME
    df_product = df[['DATA', 'GRUPO_PRODUTO', 'QUANT_PER_TIME_STEP', 'INTERVAL']]
    df_product = df_product.dropna().reset_index(drop=True)
    df_product = df_product.drop_duplicates().reset_index(drop=True)
    df_product = df_product.sort_values(by='DATA', ascending=True).reset_index(drop=True)
    df_product['INTERVAL'] = df_product['INTERVAL'].apply(to_date)

    plot = df_product[['INTERVAL', 'QUANT_PER_TIME_STEP']]
    plot = plot.dropna().reset_index(drop=True)
    plot = plot.drop_duplicates().reset_index(drop=True)
    plot = plot.sort_values(by='INTERVAL', ascending=False).reset_index(drop=True)

    return df_product, plot


def plot_product_over_time(df, product, step):
    import plotly.graph_objects as go
    
    line = go.Figure(go.Scatter(
        x=plot['INTERVAL'],
        y=plot['QUANT_PER_TIME_STEP'],
        mode='lines',
        line=dict(color='rgb(114, 172, 77)', width=3),
        ))
    line.layout.update(title=dict(text="SALES PER " + step.upper()), title_font = dict(size=18),
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title="TIME"),
                    yaxis=dict(title="UNITS SOLD"),
                    width = 700, height=500)
    line.add_annotation(
                    text=str(product.upper()),
                    xref='paper', yref='paper',
                    x=0.7, y=1.15,  # Adjust the x and y values to position the subtitle
                    showarrow=False,
                    font=dict(size=12, color='black'))
    line.layout.update(margin=dict(l=10, r=10, b=50, t=150))
    line.update_xaxes(showline=True, linewidth=1, linecolor='rgba(169, 169, 169, 1)')
    line.show()    


def define_time(df):
    from datetime import date
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta

    def retrieve_month(data):
        return data.month
    for i, data in enumerate(df['DATA']):
        df.loc[i, 'year'] = data.isocalendar().year
        df.loc[i, 'week'] = data.isocalendar().week
        df.loc[i, 'weekday'] = data.isocalendar().weekday
        d = 5 - data.isocalendar().weekday
        days_to_friday = timedelta(days=d)
        df.loc[i, 'last_day_week'] = data + days_to_friday
        df.loc[i, 'last_day_month'] = date(data.year, data.month, 1) + relativedelta(months=1) - timedelta(days=1)

    df['year'] = df['year'].astype(int)
    df['month'] = df['DATA'].apply(retrieve_month).astype(int)
    df['week'] = df['week'].astype(int)
    df['weekday'] = df['weekday'].astype(int)
    return df


def product_over_year(df_vendas, product, step, color):
    if product == 'TODOS' or product == 'ALL':
        df_date = df_vendas.copy().reset_index(drop=True)
    elif product == 'SUPRIMENTO' or product == 'MATERIAL_DE_ESCRITORIO' or product == 'EQUIPAMENTO':
        df_date = df_vendas[df_vendas['CATEGORIA']==product].copy().reset_index(drop=True)
    else:
        df_date = df_vendas[df_vendas['GRUPO_PRODUTO']==product].copy().reset_index(drop=True)
    
    if color:
        df_date = df_date[df_date['COR']==color].copy().reset_index(drop=True)
        

    df_date = define_time(df_date)

    if step == 'week' or step == 'month':
        step_column_name = 'last_day_' + step
    elif step == 'day':
        step_column_name = 'DATA'

    sales_per_time_step = df_date.groupby([step_column_name]).agg(sum=('QUANT','sum')).reset_index()
    df_date = pd.merge(df_date, sales_per_time_step, on=step_column_name, how='outer')
    quant_per_timestep = 'quant_per_' + step
    df_date.rename(columns={'sum': quant_per_timestep}, inplace=True)

    df_product = df_date[[step_column_name, quant_per_timestep]]
    df_product = df_product.dropna().reset_index(drop=True)
    df_product = df_product.drop_duplicates().reset_index(drop=True)
    df_product = df_product.sort_values(by=step_column_name, ascending=True).reset_index(drop=True)
    return df_product, df_date


def revenue_over_year(df_vendas, product, step):
    if product == 'TODOS' or product == 'ALL':
        df_date = df_vendas.copy().reset_index(drop=True)
    elif product == 'SUPRIMENTO' or product == 'MATERIAL_DE_ESCRITORIO' or product == 'EQUIPAMENTO':
        df_date = df_vendas[df_vendas['CATEGORIA']==product].copy().reset_index(drop=True)
    else:
        df_date = df_vendas[df_vendas['GRUPO_PRODUTO']==product].copy().reset_index(drop=True)
        

    df_date = define_time(df_date)

    if step == 'week' or step == 'month':
        step_column_name = 'last_day_' + step
    elif step == 'day':
        step_column_name = 'DATA'

    sales_per_time_step = df_date.groupby([step_column_name]).agg(sum=('TOTAL','sum')).reset_index()
    df_date = pd.merge(df_date, sales_per_time_step, on=step_column_name, how='outer')
    revenue_per_timestep = 'revenue_per_' + step
    df_date.rename(columns={'sum': revenue_per_timestep}, inplace=True)

    df_revenue = df_date[[step_column_name, revenue_per_timestep]]
    df_revenue = df_revenue.dropna().reset_index(drop=True)
    df_revenue = df_revenue.drop_duplicates().reset_index(drop=True)
    df_revenue = df_revenue.sort_values(by=step_column_name, ascending=True).reset_index(drop=True)
    return df_revenue


def plot_product_over_year(df_product, product, step):
    import plotly.graph_objects as go

    line = go.Figure(go.Scatter(
        x=df_product.iloc[:, 0],
        y=df_product.iloc[:, 1],
        mode='lines',
        line=dict(color='rgb(114, 172, 77)', width=3),
        ))

    line.layout.update(title=dict(text="SALES PER " + step.upper(),
                                x=0.1, y=0.9), 
                    title_font = dict(size=18),
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(title="UNITS SOLD"),
                    width = 700, height=500)
    line.add_annotation(
                    text=str(product.upper()),
                    xref='paper', yref='paper',
                    x=0.02, y=1.2,  # Adjust the x and y values to position the subtitle
                    showarrow=False,
                    font=dict(size=15, color='rgb(114, 172, 77)'))
    line.layout.update(margin=dict(l=10, r=10, b=50, t=150))
    line.update_xaxes(showline=True, linewidth=1, linecolor='rgba(169, 169, 169, 1)')

    line.show() 

def plot_over_year(list_of_dfs, product, step):
    import plotly.graph_objects as go
    
    traces = []
    colors = ['rgba(100, 159, 237, 110)', 'rgb(114, 172, 77)', 'rgba(169, 169, 169, 1)', 'rgba(100, 159, 237, 1)']

    for i, df in enumerate(list_of_dfs):
        traces.append(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], mode='lines', line=dict(color=colors[i], width=3)))
    
    fig = go.Figure(data=[trace for trace in traces])

    fig.layout.update(title=dict(text="SALES PER " + step.upper(),
                                x=0.1, y=0.9), 
                    title_font = dict(size=18),
                    plot_bgcolor='rgba(0,0,0,0)',
                    width = 700, height=500)
    fig.add_annotation(
                    text=str(product.upper()),
                    xref='paper', yref='paper',
                    x=0.02, y=1.2,  # Adjust the x and y values to position the subtitle
                    showarrow=False,
                    font=dict(size=15, color='rgb(114, 172, 77)'))
    fig.add_annotation(
                    text='# OF TOTAL <span style="color:rgba(100, 159, 237, 110);">UNITS SOLD</span> | TOTAL <span style="color:rgb(114, 172, 77);">REVENUE</span>',
                    xref='paper', yref='paper',
                    x=0.02, y=1.1,  # Adjust the x and y values to position the subtitle
                    showarrow=False,
                    font=dict(size=15, color='rgba(169, 169, 169, 1)'))
    fig.layout.update(margin=dict(l=10, r=10, b=50, t=150))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='rgba(169, 169, 169, 1)')

    fig.show() 