import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import psycopg2 as pg
from psycopg2.extras import execute_values

from config_sens import *

pd.set_option('display.max_columns', None)

warnings.filterwarnings('ignore')

PARALLEL = 5
dests_table = 'dests_to_analyze'

columns_to_agg = [
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'balanceChngOrig',
    'balanceChngDest',
    'balanceDelta',
    'delta_orig_chng_ratio',
    'delta_dest_chng_ratio'
]

win_sizes = [7, 14, 28, 90, 180]


db = pg.connect(host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password)


def custom_aggregations(dest):
    try:
        result = {}
        group = df.loc[dest]
        group = group.reset_index()
        group = group.set_index('step').sort_index()

        for col in columns_to_agg:
            for win_size in win_sizes:
                def custom_rolling_steps(series, window_size):
                    idx = series.index
                    result = []
                    for i in idx:
                        window = series[(series.index <= i) & (series.index > i - window_size)]
                        result.append(window)
                    return result

                rolling_windows = custom_rolling_steps(group[col], win_size)

                result[f'{col}_{win_size}_sum'] = pd.Series([w.sum() for w in rolling_windows], index=group.index)
                result[f'{col}_{win_size}_avg'] = pd.Series([w.mean() for w in rolling_windows], index=group.index)
                result[f'{col}_{win_size}_std'] = pd.Series([w.std() for w in rolling_windows], index=group.index)
                result[f'{col}_{win_size}_count'] = pd.Series([w.count() for w in rolling_windows], index=group.index)

                mean = result[f'{col}_{win_size}_avg']
                std = result[f'{col}_{win_size}_std']
                result[f'{col}_{win_size}_ma_diff_std'] = (group[col] - mean) / std

                result[f'{col}_{win_size}_xtreme_cnt_90'] = pd.Series([
                    np.sum(w.values > w.quantile(0.9)) if not w.empty else 0
                    for w in rolling_windows
                ], index=group.index)

                result[f'{col}_{win_size}_xtreme_cnt_10'] = pd.Series([
                    np.sum(w.values < w.quantile(0.1)) if not w.empty else 0
                    for w in rolling_windows
                ], index=group.index)
        res = pd.DataFrame(result, index=group.index)
        res = pd.concat((group, res), axis=1)
        res = res.reset_index()

        with db.cursor() as cursor:
            columns = res.columns.tolist()
            values = [tuple(row) for row in res.to_numpy()]

            query = f"INSERT INTO fraud_dataframe ({','.join(columns)}) VALUES %s ON CONFLICT DO NOTHING"
            execute_values(cursor, query, values)
            cursor.execute(f"DELETE FROM {dests_table} WHERE nameDest = %s", (dest,))

            db.commit()
            print(f"Done {dest}")

        return dest
    except Exception as e:
        print(f'EXCEPTION for {dest}: {e}')
        return None


# Load and preprocess data
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
dests_unique = sorted(pd.read_sql_query(f'select * from {dests_table}', db)['namedest'].values)
df = df[df['nameDest'].isin(dests_unique)]

df['balanceChngOrig'] = (df['newbalanceOrig'] - df['oldbalanceOrg'])
df['balanceChngDest'] = (df['newbalanceDest'] - df['oldbalanceDest'])
df['balanceDelta'] = df['balanceChngOrig'] + df['balanceChngDest']
df['delta_orig_chng_ratio'] = df['balanceDelta'] / df['balanceChngOrig']
df['delta_dest_chng_ratio'] = df['balanceDelta'] / df['balanceChngDest']
df = df.set_index('nameDest')


if __name__ == '__main__':
    print(f"To process: {len(dests_unique)} {time.time()}")

    # custom_aggregations(dests_unique[0])
    # assert False

    # Process remaining dests
    with ProcessPoolExecutor(max_workers=PARALLEL) as executor:
        futures = {
            executor.submit(custom_aggregations, dest): dest
            for dest in dests_unique
        }
        for future in as_completed(futures):
            dest_slice = futures[future]
            try:
                result = future.result()
                if result is not None:
                    print(f"Completed processing for {dest_slice}")
                else:
                    print(f"Failed processing for {dest_slice}: Result is None")
            except Exception as e:
                print(f"Failed slice {dest_slice}: {e}")
