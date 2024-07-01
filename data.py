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
    #selected_columns = ['Year_LBAMIP', 'DoY_LBAMIP', 'Hour_LBAMIP', 'NEEnogap_5day_sco2_ust']
    selected_columns = ['Year_LBAMIP', 'DoY_LBAMIP', 'Hour_LBAMIP', 'GEP_5day_sco2_ust']
    data_dict = {}
    lst = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    i_date = '2000-01-01'
    f_date = '2007-01-01'
    scale = 500
    lst = lst.select('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'SR_CLOUD_QA', 'SR_ATMOS_OPACITY', 'ST_B6').filterDate(i_date, f_date)
    
    pml = ee.ImageCollection("CAS/IGSNRR/PML/V2_v017")
    pml = pml.select('Ec','Es','Ei').filterDate(i_date, f_date)
    
    lst_raw = ee.ImageCollection('LANDSAT/LE07/C02/T1')
    lst_raw = lst_raw.select('B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6_VCID_1').filterDate(i_date, f_date)
    
    def add_year_and_julian_day(df):
        df['date_str'] = df['id'].str.split('_').str[-1]
        df['date'] = pd.to_datetime(df['date_str'], format='%Y%m%d')
        df = df.drop(columns=['date_str'])
    
        return df
    
    def add_date_column(df):
        # Ensure the 'id' column exists
        if 'id' not in df.columns:
            raise ValueError("DataFrame must contain an 'id' column")
        
        # Convert the 'id' column to datetime and add as new 'date' column
        df['date'] = pd.to_datetime(df['id'], format='%Y-%m-%d')
        
        return df

    for tower in Towers:
        if tower == 'K67':
            poi = ee.Geometry.Point(-54.959, -2.857)
            poi_buffer = poi.buffer(150).bounds()
        elif tower == 'K77':
            poi = ee.Geometry.Point(-54.8885, -3.0202)
            poi_buffer = poi.buffer(150).bounds()
        elif tower == 'K83':
            poi = ee.Geometry.Point(-54.9707, -3.017)
            poi_buffer = poi.buffer(150).bounds()
        elif tower == 'K34':
            poi = ee.Geometry.Point(-60.2091, -2.5)
        elif tower == 'RJA':
            poi = ee.Geometry.Point(-61.9331, -10.078)
        file_path = file_head + tower + file_tail
        ec_df = read_tsv_to_dataframe(file_path, selected_columns)
        #ec_df.rename(columns={'Hour_LBAMIP': 'Hour', 'NEEnogap_5day_sco2_ust': 'NEE'}, inplace=True)
        ec_df.rename(columns={'Hour_LBAMIP': 'Hour', 'GEP_5day_sco2_ust': 'NEE'}, inplace=True)
        
        lst_poi = lst.getRegion(poi, scale).getInfo()
        sat_df = pd.DataFrame(lst_poi[1:], columns=lst_poi[0])
        sat_df = add_year_and_julian_day(sat_df)       
        for column in ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7']:
            sat_df[column] = sat_df[column].astype(float) * 2.75e-05
        sat_df['ST_B6'] = sat_df['ST_B6'].astype(float) * 0.00341802
        sat_df['SR_ATMOS_OPACITY'] = sat_df['SR_ATMOS_OPACITY'].astype(float) * 0.001
        sat_df.drop(sat_df[sat_df['SR_ATMOS_OPACITY'] >= 0.3].index, inplace=True)
        sat_df['NDVI'] = (sat_df['SR_B4'].astype(float) - sat_df['SR_B3'].astype(float)) / (sat_df['SR_B4'].astype(float) + sat_df['SR_B3'].astype(float))
        sat_df['EVI'] = 2.5 * (sat_df['SR_B4'].astype(float) - sat_df['SR_B3'].astype(float)) / (sat_df['SR_B4'].astype(float) + 2.4 * sat_df['SR_B3'].astype(float) + 1)
        
        lst_raw_poi = lst_raw.getRegion(poi, scale).getInfo()
        sat_raw_df = pd.DataFrame(lst_raw_poi[1:], columns=lst_raw_poi[0])
        sat_raw_df = add_year_and_julian_day(sat_raw_df)  
        
        pml_poi = pml.getRegion(poi_buffer, scale).getInfo()
        pml_df = pd.DataFrame(pml_poi[1:], columns=pml_poi[0])
        pml_df = add_date_column(pml_df)
        
        df = pd.merge(ec_df, sat_df, on='date', how='inner')
        #df = pd.merge(df, sat_raw_df, on='date', how='inner')
        #df = pd.merge(ec_df, pml_df, on='date', how='inner')
        #df = pd.merge(df, pml_df, on='date', how='inner')
        '''
        df['D1'] = df['B1'].astype(float) - df['SR_B1'].astype(float)
        df['D2'] = df['B2'].astype(float) - df['SR_B2'].astype(float)
        df['D3'] = df['B3'].astype(float) - df['SR_B3'].astype(float)
        df['D4'] = df['B4'].astype(float) - df['SR_B4'].astype(float)
        df['D5'] = df['B5'].astype(float) - df['SR_B5'].astype(float)
        df['D7'] = df['B7'].astype(float) - df['SR_B7'].astype(float)
        '''
        
        data_dict[tower] = df
    
    return data_dict