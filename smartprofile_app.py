# 🤖 SmartProfile Marketing AI - Version Corrigée Sans Erreurs
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# Configuration
st.set_page_config(
    page_title="SmartProfile Marketing AI",
    page_icon="🤖",
    layout="wide"
)

# CSS Moderne
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

.chat-user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 20px 20px 5px 20px;
    margin: 1rem 0;
    margin-left: 20%;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.chat-assistant {
    background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 20px 20px 20px 5px;
    margin: 1rem 0;
    margin-right: 20%;
    box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.status-badge {
    background: #2ecc71;
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: bold;
}

.api-status {
    background: #f39c12;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

class SmartProfileAgent:
    """Agent conversationnel SmartProfile simplifié et corrigé"""
    
    def __init__(self):
        self.load_data()
        self.api_available = False
    
    def load_data(self):
        """Charge des données SmartProfile corrigées"""
        # Données clients avec noms de colonnes cohérents
        np.random.seed(42)
        n_customers = 1000
        
        self.customers = pd.DataFrame({
            'customer_id': [f"SP_{i:05d}" for i in range(n_customers)],
            'sector': np.random.choice(['Banque', 'Assurance', 'Tourisme', 'Presse', 'Retail'], n_customers),
            'satisfaction': np.random.normal(3.7, 0.9, n_customers),  # Nom simplifié
            'churn_risk': np.random.beta(2, 8, n_customers),
            'lifetime_value': np.random.exponential(3000, n_customers),
            'interactions': np.random.poisson(12, n_customers),
            'email_opens': np.random.poisson(6, n_customers),
            'clicks': np.random.poisson(2, n_customers),
            'channel': np.random.choice(['Email', 'SMS', 'Web', 'Mobile'], n_customers)
        })
        
        # Nettoyage des données
        self.customers['satisfaction'] = self.customers['satisfaction'].clip(1, 5)
        self.customers['churn_risk'] = self.customers['churn_risk'].clip(0, 1)
        
        # Données campagnes simplifiées
        n_campaigns = 100
        self.campaigns = pd.DataFrame({
            'campaign_id': [f"CAMP_{i:04d}" for i in range(n_campaigns)],
            'channel': np.random.choice(['Email', 'SMS', 'Web', 'Mobile', 'Social'], n_campaigns),
            'roi': np.random.uniform(50, 200, n_campaigns),
            'revenue': np.random.uniform(5000, 150000, n_campaigns),
            'cost': np.random.uniform(1000, 25000, n_campaigns),
            'status': np.random.choice(['Active', 'Completed', 'Paused'], n_campaigns, p=[0.3, 0.6, 0.1])
        })
        
        # Données interactions
        n_interactions = 2000
        self.interactions = pd.DataFrame({
            'interaction_id': [f"INT_{i:06d}" for i in range(n_interactions)],
            'sentiment': np.random.choice(['positif', 'neutre', 'négatif'], n_interactions, p=[0.6, 0.25, 0.15]),
            'channel': np.random.choice(['Email', 'SMS', 'Web', 'Mobile'], n_interactions),
            'rating': np.random.randint(1, 6, n_interactions)
        })
    
    def ask_chatgpt(self, question):
        """Génère une réponse intelligente"""
        return self.generate_smart_response(question)
    
    def generate_smart_response(self, question):
        """Génère une réponse basée sur la question"""
        q = question.lower()
        
        if any(word in q for word in ['satisfaction', 'sentiment', 'avis', 'nps']):
            return self.analyze_satisfaction()
        elif any(word in q for word in ['campagne', 'email', 'sms', 'performance', 'roi']):
            return self.analyze_campaigns()
        elif any(word in q for word in ['churn', 'rétention', 'risque', 'partir']):
            return self.analyze_churn()
        elif any(word in q for word in ['recommandation', 'conseil', 'action', 'améliorer']):
            return self.generate_recommendations()
        else:
            return self.help_response()
    
    def analyze_satisfaction(self):
        """Analyse de satisfaction simplifiée"""
        avg_satisfaction = self.customers['satisfaction'].mean()
        nps = self.calculate_nps()
        high_satisfaction = (self.customers['satisfaction'] >= 4.5).sum()
        
        response = f"""📊 **Analyse Satisfaction Client SmartProfile**

😊 **Score moyen**: {avg_satisfaction:.1f}/5 
📈 **Net Promoter Score**: {nps:+.1f} {"🟢" if nps > 50 else "🟡" if nps > 0 else "🔴"}
⭐ **Clients très satisfaits**: {high_satisfaction} ({high_satisfaction/len(self.customers)*100:.1f}%)

🔍 **Par secteur**:
{self.get_satisfaction_by_sector()}

🎯 **Actions recommandées**:
• Programme de fidélisation pour {high_satisfaction} clients promoteurs
• Plan d'amélioration pour secteurs sous-performants
• Personnalisation par niveau de satisfaction

💰 **Impact estimé**: +15% rétention, +€{avg_satisfaction*50000:.0f} revenus annuels"""

        # Graphique
        fig = px.histogram(self.customers, x='satisfaction', nbins=10,
                          title="Distribution Satisfaction Client")
        st.plotly_chart(fig, use_container_width=True)
        
        return response
    
    def analyze_campaigns(self):
        """Analyse des campagnes"""
        avg_roi = self.campaigns['roi'].mean()
        best_channel = self.campaigns.groupby('channel')['roi'].mean().idxmax()
        total_revenue = self.campaigns['revenue'].sum()
        
        response = f"""📧 **Performance Campagnes SmartProfile**

📈 **ROI moyen**: {avg_roi:.1f}%
🏆 **Canal star**: {best_channel} 
💰 **Revenus générés**: €{total_revenue:,.0f}
📊 **Campagnes actives**: {len(self.campaigns[self.campaigns['status']=='Active'])}

🎯 **Top canaux**:
{self.get_top_channels()}

🚀 **Optimisations**:
• Augmenter budget {best_channel}
• A/B tester créatifs sous-performants
• Automatisation cross-canal

💡 **Potentiel**: +25% ROI avec optimisation IA"""

        # Graphique
        channel_perf = self.campaigns.groupby('channel')['roi'].mean()
        fig = px.bar(x=channel_perf.index, y=channel_perf.values,
                    title="ROI par Canal Marketing")
        st.plotly_chart(fig, use_container_width=True)
        
        return response
    
    def analyze_churn(self):
        """Analyse du churn"""
        high_risk = (self.customers['churn_risk'] > 0.7).sum()
        medium_risk = ((self.customers['churn_risk'] > 0.4) & (self.customers['churn_risk'] <= 0.7)).sum()
        avg_clv = self.customers['lifetime_value'].mean()
        
        response = f"""🔮 **Analyse Risque de Churn SmartProfile**

⚠️ **Clients à risque élevé**: {high_risk} ({high_risk/len(self.customers)*100:.1f}%)
🟡 **Risque modéré**: {medium_risk} ({medium_risk/len(self.customers)*100:.1f}%)
💎 **CLV moyen**: €{avg_clv:.0f}
💸 **Valeur à risque**: €{high_risk * avg_clv * 0.6:.0f}

🔍 **Profils à risque**:
• Secteur le plus touché: {self.get_churn_by_sector()}
• Engagement faible: <{self.customers['interactions'].quantile(0.25):.0f} interactions
• Satisfaction: <{self.customers['satisfaction'].quantile(0.3):.1f}/5

🎯 **Stratégie de rétention**:
• Campagne ciblée {high_risk} clients à risque
• Programme win-back automatisé
• Amélioration expérience produit

💰 **ROI rétention**: €{avg_clv*0.7:.0f} par client sauvé"""

        # Graphique
        fig = px.histogram(self.customers, x='churn_risk', nbins=20,
                          title="Distribution Risque de Churn")
        st.plotly_chart(fig, use_container_width=True)
        
        return response
    
    def generate_recommendations(self):
        """Recommandations stratégiques"""
        best_channel = self.campaigns.groupby('channel')['roi'].mean().idxmax()
        worst_sector = self.customers.groupby('sector')['satisfaction'].mean().idxmin()
        
        response = f"""🎯 **Recommandations Stratégiques SmartProfile**

📈 **Priorités Q1 2024**:
• Optimiser canal {best_channel} (meilleur ROI)
• Plan d'action secteur {worst_sector} (satisfaction critique)
• Automatisation séquences de nurturing

🚀 **Initiatives Q2-Q3**:
• IA prédictive temps réel
• Personnalisation omnicanale avancée
• Expansion segments high-value

🔮 **Vision 2024**:
• Agents conversationnels comme cette démo
• Prédiction lifetime value temps réel
• Orchestration parcours client IA

💰 **Business Case**:
• Revenus: +€{self.campaigns['revenue'].sum()*0.3:.0f} (+30%)
• Rétention: +25%
• Efficacité: +40%

🎯 **ROI global estimé**: 250% sur 12 mois"""
        
        return response
    
    # Méthodes utilitaires simplifiées
    def calculate_nps(self):
        promoters = (self.customers['satisfaction'] >= 4.5).sum()
        detractors = (self.customers['satisfaction'] <= 2.5).sum()
        return (promoters - detractors) / len(self.customers) * 100
    
    def get_satisfaction_by_sector(self):
        result = ""
        for sector, satisfaction in self.customers.groupby('sector')['satisfaction'].mean().sort_values(ascending=False).items():
            result += f"• {sector}: {satisfaction:.1f}/5\n"
        return result
    
    def get_top_channels(self):
        result = ""
        for channel, roi in self.campaigns.groupby('channel')['roi'].mean().sort_values(ascending=False).head(3).items():
            result += f"• {channel}: {roi:.1f}% ROI\n"
        return result
    
    def get_churn_by_sector(self):
        return self.customers.groupby('sector')['churn_risk'].mean().idxmax()
    
    def help_response(self):
        return f"""🤖 **SmartProfile Marketing Intelligence**

Je suis votre expert IA pour l'analyse marketing SmartProfile.

📊 **Mes capacités**:
• Analyse satisfaction et NPS client
• Performance campagnes multi-canal  
• Prédiction et prévention churn
• Recommandations ROI optimisées

💡 **Questions types**:
• "Analyse la satisfaction de nos clients bancaires"
• "Compare la performance Email vs SMS"  
• "Identifie les clients à risque de churn"
• "Recommande des actions marketing"

📈 **Données disponibles**: 
{len(self.customers):,} clients • {len(self.campaigns)} campagnes • {len(self.interactions):,} interactions

🎯 **Mode**: 🤖 Simulation IA SmartProfile"""

def main():
    """Interface principale"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 SmartProfile Marketing AI</h1>
        <p>Assistant Conversationnel Intelligence Marketing</p>
        <small>Customer Data Platform • Modèles Prédictifs • Interface Naturelle</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation de l'agent
    if 'agent' not in st.session_state:
        with st.spinner("🚀 Initialisation SmartProfile AI..."):
            st.session_state.agent = SmartProfileAgent()
            st.session_state.messages = []
        st.success("✅ Agent SmartProfile activé !")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="status-badge">SmartProfile AI Active</div>', unsafe_allow_html=True)
        st.markdown('<div class="api-status">🤖 Mode Simulation</div>', unsafe_allow_html=True)
        
        # Métriques live
        st.subheader("📊 Métriques Live")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👥 Clients", f"{len(st.session_state.agent.customers):,}")
            st.metric("📧 Campagnes", len(st.session_state.agent.campaigns))
        with col2:
            avg_satisfaction = st.session_state.agent.customers['satisfaction'].mean()
            st.metric("😊 Satisfaction", f"{avg_satisfaction:.1f}/5")
            avg_roi = st.session_state.agent.campaigns['roi'].mean()
            st.metric("📈 ROI", f"{avg_roi:.1f}%")
        
        # Questions prédéfinies
        st.subheader("💡 Questions Intelligentes")
        questions = [
            "Analyse la satisfaction client par secteur",
            "Performance des campagnes email vs SMS",
            "Identifie les clients à risque de churn",
            "Recommande des actions marketing",
            "Optimise l'allocation budget"
        ]
        
        for question in questions:
            if st.button(question, key=f"q_{hash(question)}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("🧠 Analyse en cours..."):
                    response = st.session_state.agent.ask_chatgpt(question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Dashboard métriques principales
    st.subheader("📊 Dashboard SmartProfile")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(st.session_state.agent.customers)
        st.markdown(f'<div class="metric-card"><h3>👥 Clients Actifs</h3><h2>{total_customers:,}</h2><small>+5.2% vs mois dernier</small></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        active_campaigns = len(st.session_state.agent.campaigns[st.session_state.agent.campaigns['status']=='Active'])
        st.markdown(f'<div class="metric-card"><h3>📧 Campagnes Live</h3><h2>{active_campaigns}</h2><small>ROI: 127%</small></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        nps = st.session_state.agent.calculate_nps()
        st.markdown(f'<div class="metric-card"><h3>📈 NPS Score</h3><h2>{nps:+.0f}</h2><small>{"Excellent" if nps > 50 else "Bon" if nps > 0 else "Critique"}</small></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        total_revenue = st.session_state.agent.campaigns['revenue'].sum()
        st.markdown(f'<div class="metric-card"><h3>💰 Revenus</h3><h2>€{total_revenue/1000:.0f}K</h2><small>+12% vs objectif</small></div>', 
                   unsafe_allow_html=True)
    
    # Zone de chat
    st.subheader("💬 Chat avec votre Expert SmartProfile")
    
    # Affichage des messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-user"><strong>👤 Vous :</strong> {message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant"><strong>🤖 SmartProfile AI :</strong><br>{message["content"]}</div>', 
                       unsafe_allow_html=True)
    
    # Input utilisateur
    user_input = st.chat_input("Posez votre question marketing...")
    
    if user_input:
        # Ajouter message utilisateur
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Créer un conteneur pour la réponse en cours
        response_container = st.empty()
        
        # Générer réponse
        with st.spinner("🧠 Analyse par l'IA SmartProfile..."):
            time.sleep(1)
            full_response = st.session_state.agent.ask_chatgpt(user_input)
            
            # Simuler l'écriture progressive comme ChatGPT
            partial_response = ""
            for char in full_response:
                partial_response += char
                response_container.markdown(
                    f'<div class="chat-assistant"><strong>🤖 SmartProfile AI :</strong><br>{partial_response}</div>',
                    unsafe_allow_html=True
                )
                time.sleep(0.01)  # Ajuster la vitesse de frappe
        
        # Ajouter la réponse complète à l'historique
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

if __name__ == "__main__":
    main()