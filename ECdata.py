import pandas as pd

def read_tsv_to_dataframe(file_path, selected_columns=None):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split('\t')

    data = []

    for line in lines[3:]:
        fields = line.strip().split('\t')
        data.append(fields[:211])

    df = pd.DataFrame(data, columns=header)
    
    if selected_columns!=None:
        df = df[selected_columns]
        
    return df

def ECdata(Towers=['K67', 'K77', 'K83']):
    file_head = 'CD32_Fluxes_Brazil_1842/data/'
    file_tail = 'day_CfluxBF.txt'
    selected_columns = ['Year_LBAMIP', 'DoY_LBAMIP', 'Hour_LBAMIP', 'NEEnogap_5day_sco2_ust']
    data_array = {}

    for tower in Towers:
        file_path = file_head + tower + file_tail
        df = read_tsv_to_dataframe(file_path, selected_columns)
        data_array[tower] = df
    
    return data_array