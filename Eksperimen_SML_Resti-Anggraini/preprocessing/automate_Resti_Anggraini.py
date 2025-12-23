import pandas as pd
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def run_automation():
    config = load_config('preprocessing.yml')
    raw_path = config['dataset']['raw_path']
    final_path = config['dataset']['final_path']
    params = config['params']

    try:
        df = pd.read_csv(raw_path, encoding='ISO-8859-1')
    except FileNotFoundError:
        return

    df_clean = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    df_clean = df_clean[
        df_clean['Quantity'] < params['max_quantity']
    ]
    df_clean.dropna(subset=['CustomerID'], inplace=True)
    df_clean = df_clean.drop_duplicates()
    df_clean['InvoiceDate'] = pd.to_datetime(
        df_clean['InvoiceDate']
    )

    df_clean['Month_Year'] = df_clean['InvoiceDate'].dt.to_period('M')

    monthly_sales = df_clean.pivot_table(
        index='StockCode',
        columns='Month_Year',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )

    product_features = pd.DataFrame()
    product_features['Avg_Sales'] = monthly_sales.mean(axis=1)
    product_features['Std_Dev'] = monthly_sales.std(axis=1)
    product_features['Max_Sales'] = monthly_sales.max(axis=1)
    product_features['CV'] = (
        product_features['Std_Dev'] / product_features['Avg_Sales']
    )
    product_features['CV'] = product_features['CV'].fillna(0)

    product_info = df_clean.groupby('StockCode').agg({
        'UnitPrice': 'mean',
        'Description': 'first'
    })

    final_data = product_features.join(product_info)
    final_data['Avg_Revenue'] = (
        final_data['Avg_Sales'] * final_data['UnitPrice']
    )

    rev_threshold = final_data['Avg_Revenue'].quantile(
        params['revenue_quantile']
    )
    max_sales_threshold = final_data['Max_Sales'].quantile(
        params['max_sales_quantile']
    )
    cv_threshold = final_data['CV'].median()

    def quadrant_labeling(row):
        is_high_revenue = (
            row['Avg_Revenue'] > rev_threshold
            or row['Max_Sales'] > max_sales_threshold
        )
        is_stable = (row['CV'] <= cv_threshold)

        if is_high_revenue and is_stable:
            return 3
        elif is_high_revenue:
            return 2
        elif is_stable:
            return 1
        else:
            return 0

    final_data['Label'] = final_data.apply(quadrant_labeling, axis=1)
    final_data.reset_index().to_csv(final_path, index=False)


if __name__ == "__main__":
    run_automation()
