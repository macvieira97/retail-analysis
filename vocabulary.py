from get_database_data import get_fabricante


#CORES
cores = ['PRETO', 'BRANCO', 'AZUL', 'OURO', 'AMARELO', 'VERMELHO', 'VERDE', 'CINZA', 'MARROM', 'LARANJA',
          'COLORIDO', 'PASTEL', 'ROXO', 'VIOLETA', 'VERDE', 'TURQUESA', 'ROYAL', 'PETROLEO', 'DOURADO', 'SILVER', 'GALVANIZADO',
          'FOSCO', 'MAGENTA', 'NIQUELADO', 'VERDE LIMAO', 'BRILHO', 'CRISTAL', 'MARINHO', 'BANDEIRA', 'METALIZADO', 'ESF', 'PET', 'SEMI']
#MARCAS
fab = get_fabricante()
#fab = fab.drop(fab[fab['ELIMINADO'] == 'S'].index).reset_index(drop=True)
marcas = fab['FABRICANTE'].to_list() + ['SAMSUNG','HP', 'KYOCERA', 'MITA', 'RICOH', 'RICOH AF', 'RICOH AF.' 'BROTHER', 'SHARP', 'CANON', 'TOSHIBA', 'MINOLTA', 'KONICA']

#PREPOSICOES
preposicoes = ['a', 'ante', 'apos', 'ate', 'com', 'contra', 'de', 'desde', 'em', 'entre', 'para', 'per', 
               'perante', 'por', 'sem', 'sob', 'sobre', 'tras', 'da', 'do', 'e', 'ACO', 'UV']
preposicoes = [word.upper() for word in preposicoes]

# ABBREVIATIONS
abb = {'C/': 'COM', 'P/': 'PARA', 'CAR': 'CARTUCHO', 'CAR.': 'CARTUCHO', 'CART.': 'CARTUCHO', 'CART': 'CARTUCHO', 'MOD': 'MODELO', 
       'PLAST.': 'PLASTICO', 'SUP': 'SUPERIOR', 'SUP.': 'SUPERIOR', 'INF': 'INFERIOR', 'INF.': 'INFERIOR'}

# MODELS
models = ['A3', 'A4', 'A5', 'A6', 'OFICIO', 'SULFITE', 'MOD', 'MOD.', 'MODELO', 'CARTUCHO', 'CART.', 'CAR.', 'CAR', 'CART', 'XEROX', 'PREMIUM', 'COMPATIVEL',
          'CARTTK', 'REFIL', 'AF', 'BAG', 'LINE', 'PVC', 'ACETATO', 'SEMI', 'PROMO', 'DUPLAMAX', 'UN', 'STD', 'STANDARD', 'CLASSIC',  'SPID LASER']

# CATEGORIES
equipamentos_unigrams = ['Canteadeira', 'Contador', 'cedulas', 'moedas', 'cedula', 'Cortador', 'Desumidificador', 'Encadernadora', 'Cortadeira',
                'Fragmentadora', 'Gaveta', 'Guilhotina', 'Plastificadora', 'Refiladora', 'Rotulador', 'Seladora', 'Serrilhadeira', 'Termolaminadora', 
                'Transformador', 'Vincadeira', 'Motor', 'Solda', 'furador']
equipamentos_bigrams = ['Plastificadora Rolo']
equipamentos_unigrams = [word.upper() for word in equipamentos_unigrams]
equipamentos_bigrams = [word.upper() for word in equipamentos_bigrams]

material_escritorio_unigrams = ['Perfurador', 'Bobina ECF', 'Calculadora', 'Elastico','Caneta', 'Claviculario', 'carbono', 'Clips', 'Clip', 
                                'Alfinete', 'Cola', 'Envelope', 'Estilete', 'Estojo', 'Etiquetador', 'Etiqueta','Fita','Grampeador', 'Grampo',
                                'Lamina', 'Maleta', 'Etiquetadora', 'Apagador', 'Marcador', 'Mouse', 'Numerador', 'Papel', 'Pasta', 'Pendriver', 
                                'Perfurador', 'Recibo', 'Pistola', 'Protetor', 'acrilico', 'Removedor','Regua', 'Teclado', 'Tesoura', 'discos']
material_escritorio_bigrams = ['quadro branco', 'Bloco Anotacoes', 'Fita Adesiva', 'Manta ima', 'Suporte fita', 'Fita Corretiva', 'Marca texto', 
                               'Fita Rotulador']
material_escritorio_unigrams = [word.upper() for word in material_escritorio_unigrams]
material_escritorio_bigrams = [word.upper() for word in material_escritorio_bigrams]

suprimentos_unigrams = ['PAPEL CELOFANE', 'SACO METALIZADO', 'PULSEIRA', 'TYVEK', 'TINTA', 'CRACHA', 'CHIP', 'CARTAO', 'PLACA', 'ALICATE', 'TONER', 'WIRE-O', 
               'GARRA', 'ESPIRAL','CANALETA', 'CAPA', 'PET', 'BOPP', 'SPEED', 'BOBINA', 'BOBINAS', 'POLASEAL', 'CARREGADOR', 'CARTUCHO', 'ROLO', 
               'CILINDRO',  'CARRIER', 'ROLOS']
suprimentos_bigrams = ['porta cartao', 'porta cracha', 'porta registro', 'porta cedulas', 'porta cnh', 'porta cpf', 'porta rg', 'porta titulo',
                'porta carteira', 'porta inps', 'porta ipva', 'CARRIER IDENTIDADE', 'CLIPS CRACHA', 'CLIP CRACHA', 'CORDAO CRACHA', 
                'POWER BANK', 'PET CRACHA', 'capa pp', 'protetor cracha', 'ROLOS OFICIO'] 
suprimentos_unigrams =  [word.upper() for word in suprimentos_unigrams]
suprimentos_bigrams =  [word.upper() for word in suprimentos_bigrams]


categories = {'equipamentos': {'unigrams' : equipamentos_unigrams, 'bigrams': equipamentos_bigrams}, 
              'material_de_escritorio': {'unigrams': material_escritorio_unigrams, 'bigrams': material_escritorio_bigrams}, 
              'suprimentos': {'unigrams': suprimentos_unigrams, 'bigrams': suprimentos_bigrams}}