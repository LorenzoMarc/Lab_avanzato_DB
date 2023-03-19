import pandas as pd
from sqlalchemy import types, create_engine, DateTime
import time
from tqdm import tqdm
from pathlib import PurePath
import re


def extract_name(file):
    name = PurePath(file).stem
    name = name.replace('-', '_')
    name = name.replace(' ', '_')
    name = name.replace('.', '_', 1)
    name = name.replace('(', '')
    name = name.replace(')', '')
    name = name.split('.')[0]
    line = re.sub(r"-\ \.", "", name)
    return line


def set_conn_parameters(host, port, db, user, psw):

    engine = create_engine('postgresql+psycopg2://'+user.strip()+':'+psw.strip()+'@'+host.strip()+':'+port.strip()+'/'+db.strip())
    return engine


## CHECK length DBCs files (or total name file)
def upload_dfs(list_to_upload, engine):
    for df, df_name in tqdm(iterable=list_to_upload, unit='tables', total=len(list_to_upload), desc='Uploading data to DB...'):
        df.to_sql(df_name.lower(), con=engine, schema="public", if_exists='replace', index=True)
    return True


def main(csv_file, engine, df=None):
    print('Uploading to DB... ')
    start = time.time()
    table_name = csv_file.split('.')[0]

    if df is not None:

        dbc_table_name = df.split('.')[0]
        dbc_df = pd.read_csv(df)

        dbc_df.to_sql(dbc_table_name.lower(), con=engine,schema="FcaData", if_exists='replace',
                      dtype={'index':types.Integer(),'Signal_name': types.String(), 'Start_bit': types.Integer(),
                             'Length_bit': types.Integer(), 'Modality': types.String(), 'Scale': types.Float(),
                             'Offset':types.Float(), 'Min': types.Float(),'Max': types.Float(), 'Unit': types.String(),
                             'Receiver': types.String(), 'Msg_Name':types.String(), 'CAN_ID':types.String(),
                             'Length_Data': types.Integer(), 'Sender': types.String()}, index_label='index')

    data = pd.read_csv(csv_file, usecols=['Msg_Name', 'Timestamp', 'Signal_name', 'Code', 'CAN_ID','Value'],
                       dtype={'Value':'string[pyarrow]'})

    data.to_sql(table_name.lower(), con=engine, schema="FcaData", if_exists='replace',chunksize=20000,
                dtype={'Msg_Name': types.String(), 'Timestamp': DateTime(),
                       'Signal_name': types.String(), 'Code': types.String(),
                       'CAN_ID': types.String(), 'Value': types.String()} )

    print("END UPLOAD in %s" % (time.time() - start))
    print('Data Uploaded to host')
    print('Choose other files or close this application')

    return True