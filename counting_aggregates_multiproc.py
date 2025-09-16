import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sqlite3
import pickle
import warnings

warnings.filterwarnings('ignore')

PARALLEL = 10
DB_FILE = 'aggregate_dataframes.sqlite'

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


def init_db():
    """Initialize SQLite database with a results table."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                dest TEXT PRIMARY KEY,
                data BLOB
            )
        """)
        conn.commit()


def save_result(dest, df):
    """Save DataFrame result to SQLite as a pickled BLOB."""
    try:
        pickled_data = pickle.dumps(df)
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("INSERT OR REPLACE INTO results (dest, data) VALUES (?, ?)", (str(dest), pickled_data))
            conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving result for {dest}: {e}")


def get_processed_dests():
    """Retrieve processed dests and their DataFrames from SQLite."""
    results = {}
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.execute("SELECT dest, data FROM results")
            results = {row[0]: pickle.loads(row[1]) for row in cursor.fetchall()}
    except sqlite3.Error as e:
        print(f"Error reading from {DB_FILE}: {e}")
    return results


def custom_aggregations(dest):
    try:
        result = {}
        group = df.loc[dest]
        if isinstance(group, pd.Series):
            for col in columns_to_agg:
                for win_size in win_sizes:
                    result[f'{col}_{win_size}_sum'] = [group[col]]
                    result[f'{col}_{win_size}_avg'] = [group[col]]
                    result[f'{col}_{win_size}_std'] = [None]
                    result[f'{col}_{win_size}_count'] = [1]
                    result[f'{col}_{win_size}_ma_diff_std'] = [None]
                    result[f'{col}_{win_size}_xtreme_cnt_90'] = [0]
                    result[f'{col}_{win_size}_xtreme_cnt_10'] = [0]
            # group = group.to_frame().T
            res = pd.DataFrame(result)
            return res
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
        res = pd.DataFrame(result)
        print('Done', dest)
        save_result(dest, res)
        return res
    except Exception as e:
        print(f'EXCEPTION for {dest}: {e}')
        return None


# Load and preprocess data
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
v_c = df['nameDest'].value_counts()
dests_unique = v_c[v_c > 1].index.values
df = df[df['nameDest'].isin(dests_unique)]

df['balanceChngOrig'] = (df['newbalanceOrig'] - df['oldbalanceOrg'])
df['balanceChngDest'] = (df['newbalanceDest'] - df['oldbalanceDest'])
df['balanceDelta'] = df['balanceChngOrig'] + df['balanceChngDest']
df['delta_orig_chng_ratio'] = df['balanceDelta'] / df['balanceChngOrig']
df['delta_dest_chng_ratio'] = df['balanceDelta'] / df['balanceChngDest']
df = df.set_index('nameDest')


if __name__ == '__main__':
    init_db()
    processed_dests = get_processed_dests()
    dests_to_process = [dest for dest in dests_unique if str(dest) not in processed_dests]

    print(
        f"Total dests: {len(dests_unique)}, To process: {len(dests_to_process)}, Already processed: {len(processed_dests)}")

    # Process remaining dests
    with ProcessPoolExecutor(max_workers=PARALLEL) as executor:
        futures = {
            executor.submit(custom_aggregations, dest): dest
            for dest in dests_to_process
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
