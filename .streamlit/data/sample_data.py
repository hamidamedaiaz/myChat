# Générateur de données SmartProfile réalistes
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class SmartProfileDataGenerator:
    """Génère des données réalistes pour SmartProfile"""
    
    def __init__(self):
        self.sectors = ['Banque', 'Assurance', 'Tourisme', 'Presse', 'Retail', 'Services Financiers']
        self.channels = ['Email', 'SMS', 'Web Push', 'Mobile App', 'Site Web', 'Social Media']
        self.campaign_types = ['Acquisition', 'Rétention', 'Cross-sell', 'Up-sell', 'Réactivation', 'Fidélisation']
        
    def generate_customers(self, n=1000):
        """Génère des données clients"""
        np.random.seed(42)
        
        customers = pd.DataFrame({
            'customer_id': [f"SP_{i:05d}" for i in range(n)],
            'sector': np.random.choice(self.sectors, n),
            'signup_date': pd.date_range('2023-01-01', periods=n, freq='2H'),
            'revenue': np.random.exponential(1500, n),
            'interactions_total': np.random.poisson(12, n),
            'email_opens': np.random.poisson(6, n),
            'clicks': np.random.poisson(2, n),
            'purchases': np.random.poisson(0.8, n),
            'satisfaction_score': np.random.normal(3.7, 0.9, n),
            'churn_risk': np.random.beta(2, 8, n),
            'lifetime_value': np.random.exponential(3000, n),
            'preferred_channel': np.random.choice(self.channels, n),
            'last_activity': pd.date_range('2024-01-01', periods=n, freq='3H')
        })
        
        # Nettoyage des données
        customers['satisfaction_score'] = customers['satisfaction_score'].clip(1, 5)
        customers['churn_risk'] = customers['churn_risk'].clip(0, 1)
        
        return customers
    
    def generate_campaigns(self, n=100):
        """Génère des données de campagnes"""
        np.random.seed(43)
        
        campaigns = pd.DataFrame({
            'campaign_id': [f"CAMP_{i:04d}" for i in range(n)],
            'campaign_name': [f"Campagne {self.campaign_types[i%len(self.campaign_types)]} {i+1}" for i in range(n)],
            'type': np.random.choice(self.campaign_types, n),
            'channel': np.random.choice(self.channels, n),
            'target_sector': np.random.choice(self.sectors, n),
            'start_date': pd.date_range('2024-01-01', periods=n, freq='2D'),
            'sent': np.random.randint(1000, 50000, n),
            'opened': np.random.randint(100, 20000, n),
            'clicked': np.random.randint(10, 3000, n),
            'converted': np.random.randint(1, 300, n),
            'revenue_generated': np.random.uniform(5000, 150000, n),
            'cost': np.random.uniform(1000, 25000, n),
            'status': np.random.choice(['Active', 'Completed', 'Paused'], n, p=[0.3, 0.6, 0.1])
        })
        
        # Calcul des métriques
        campaigns['open_rate'] = (campaigns['opened'] / campaigns['sent'] * 100).round(2)
        campaigns['click_rate'] = (campaigns['clicked'] / campaigns['opened'] * 100).round(2)
        campaigns['conversion_rate'] = (campaigns['converted'] / campaigns['clicked'] * 100).round(2)
        campaigns['roi'] = ((campaigns['revenue_generated'] - campaigns['cost']) / campaigns['cost'] * 100).round(2)
        campaigns['cpa'] = (campaigns['cost'] / campaigns['converted']).round(2)
        
        return campaigns
    
    def generate_interactions(self, n=2000):
        """Génère des interactions clients"""
        np.random.seed(44)
        
        interactions = pd.DataFrame({
            'interaction_id': [f"INT_{i:06d}" for i in range(n)],
            'customer_id': [f"SP_{random.randint(0, 999):05d}" for _ in range(n)],
            'campaign_id': [f"CAMP_{random.randint(0, 99):04d}" for _ in range(n)],
            'interaction_type': np.random.choice(['Email Open', 'Click', 'Purchase', 'Support', 'Website'], n),
            'channel': np.random.choice(self.channels, n),
            'sentiment': np.random.choice(['positif', 'neutre', 'négatif'], n, p=[0.6, 0.25, 0.15]),
            'satisfaction_rating': np.random.randint(1, 6, n),
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='30min'),
            'device': np.random.choice(['Desktop', 'Mobile', 'Tablet'], n, p=[0.4, 0.5, 0.1]),
            'location': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice'], n)
        })
        
        return interactions