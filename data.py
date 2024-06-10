import pandas as pd
import ee
ee.Initialize()

def read_tsv_to_dataframe(file_path, selected_columns=None):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split('\t')

    data = []

    for line in lines[3:]:
        fields = line.strip().split('\t')
        data.append(fields[:211])

    df = pd.DataFrame(data, columns=header)
    
    if selected_columns is not None:
        df = df[selected_columns]
        
    # Convert Julian Day and Year to date
    df['Year'] = df['Year_LBAMIP'].astype(float).astype(int)
    df['Julian_Day'] = df['DoY_LBAMIP'].astype(float).astype(int)
    df['date'] = pd.to_datetime(df['Year'], format='%Y') + pd.to_timedelta(df['Julian_Day'] - 1, unit='D')
    
    return df

def combined_data(Towers=['K67', 'K77', 'K83']):
    file_head = 'CD32_Fluxes_Brazil_1842/data/'
    file_tail = 'day_CfluxBF.txt'
    selected_columns = ['Year_LBAMIP', 'DoY_LBAMIP', 'Hour_LBAMIP', 'NEEnogap_5day_sco2_ust']
    data_dict = {}
    lst = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    i_date = '2000-01-01'
    f_date = '2007-01-01'
    scale = 500
    lst = lst.select('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7').filterDate(i_date, f_date)
    
    def add_year_and_julian_day(df):
        df['date_str'] = df['id'].str.split('_').str[-1]
        df['date'] = pd.to_datetime(df['date_str'], format='%Y%m%d')
        df = df.drop(columns=['date_str'])
    
        return df

    for tower in Towers:
        if tower == 'K67':
            poi = ee.Geometry.Point(-54.959, -2.857)
        elif tower == 'K77':
            poi = ee.Geometry.Point(-54.8885, -3.0202)
        elif tower == 'K83':
            poi = ee.Geometry.Point(-54.9707, -3.017)
        file_path = file_head + tower + file_tail
        ec_df = read_tsv_to_dataframe(file_path, selected_columns)
        ec_df.rename(columns={'Hour_LBAMIP': 'Hour', 'NEEnogap_5day_sco2_ust': 'NEE'}, inplace=True)
        
        lst_poi = lst.getRegion(poi, scale).getInfo()
        sat_df = pd.DataFrame(lst_poi[1:], columns=lst_poi[0])
        sat_df = add_year_and_julian_day(sat_df)        
        
        df = pd.merge(ec_df, sat_df, on='date', how='inner')
        
        data_dict[tower] = df
    
    return data_dict