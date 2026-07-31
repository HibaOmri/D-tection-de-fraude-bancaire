import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(train_path='data/fraudTrain.csv', test_path='data/fraudTest.csv', output_dir='reports/figures'):
    os.makedirs(output_dir, exist_ok=True)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    fraud_train_cnt = df_train['is_fraud'].sum()
    fraud_test_cnt = df_test['is_fraud'].sum()
    fraud_train_pct = df_train['is_fraud'].mean() * 100
    fraud_test_pct = df_test['is_fraud'].mean() * 100
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x='is_fraud', data=df_train, ax=ax[0], palette=['#2b5c8f', '#d9534f'])
    ax[0].set_title(f'Train Target Balance (Fraud: {fraud_train_pct:.2f}%)')
    ax[0].set_yscale('log')
    
    sns.countplot(x='is_fraud', data=df_test, ax=ax[1], palette=['#2b5c8f', '#d9534f'])
    ax[1].set_title(f'Test Target Balance (Fraud: {fraud_test_pct:.2f}%)')
    ax[1].set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_balance.png'), dpi=300)
    plt.close()
    
    USD_TO_MAD = 10.0
    df_train['amt_mad'] = df_train['amt'] * USD_TO_MAD
    
    amt_legit = df_train[df_train['is_fraud'] == 0]['amt_mad']
    amt_fraud = df_train[df_train['is_fraud'] == 1]['amt_mad']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(np.log1p(amt_legit), label='Légitime', color='#2b5c8f', fill=True, ax=ax)
    sns.kdeplot(np.log1p(amt_fraud), label='Fraude', color='#d9534f', fill=True, ax=ax)
    ax.set_title('Distribution Log(Montant en DH + 1) : Légitime vs Fraude')
    ax.set_xlabel('Log(Montant DH + 1)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'amt_distribution.png'), dpi=300)
    plt.close()
    
    cat_summary = df_train.groupby('category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    cat_summary['mean_pct'] = cat_summary['mean'] * 100
    cat_summary = cat_summary.sort_values(by='mean_pct', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='mean_pct', y='category', data=cat_summary, palette='Reds_r', ax=ax)
    ax.set_title('Taux de Fraude par Catégorie de Commerçant (%)')
    ax.set_xlabel('Taux de Fraude (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fraud_by_category.png'), dpi=300)
    plt.close()
    
    df_train['trans_date_trans_time'] = pd.to_datetime(df_train['trans_date_trans_time'])
    df_train['hour'] = df_train['trans_date_trans_time'].dt.hour
    
    hourly_fraud = df_train.groupby('hour')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    hourly_fraud['pct'] = hourly_fraud['mean'] * 100
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x='hour', y='pct', data=hourly_fraud, marker='o', color='#d9534f', linewidth=2.5, ax=ax)
    ax.set_title('Taux de Fraude par Heure de la Journée (%)')
    ax.set_xlabel('Heure (0h - 23h)')
    ax.set_ylabel('Taux de Fraude (%)')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fraud_by_hour.png'), dpi=300)
    plt.close()
    
    return cat_summary

if __name__ == '__main__':
    run_eda()
