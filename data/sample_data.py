import pandas as pd
import numpy as np

class SmartProfileDataGenerator:
    @staticmethod
    def generate_customer_data(n=500):
        """Génère des données clients d'exemple"""
        np.random.seed(42)
        return pd.DataFrame({
            'client_id': [f"C_{i:03d}" for i in range(n)],
            'sector': np.random.choice(['Banque', 'Assurance', 'Tourisme', 'Retail'], n),
            'satisfaction': np.random.randint(1, 6, n),
            'revenue': np.random.uniform(100, 5000, n),
            'risk_churn': np.random.uniform(0, 1, n)
        })

    @staticmethod
    def generate_campaign_data(n=20):
        """Génère des données de campagnes d'exemple"""
        np.random.seed(42)
        return pd.DataFrame({
            'campagne': [f"Camp_{i}" for i in range(n)],
            'canal': np.random.choice(['Email', 'SMS', 'Web', 'Mobile'], n),
            'roi': np.random.uniform(50, 200, n),
            'taux_ouverture': np.random.uniform(10, 40, n)
        })
