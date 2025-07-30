# 🤖 SmartProfile Marketing AI - Version Hybride avec ChatGPT Automatique
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import requests
import json

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

.chat-assistant-thinking {
    background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 20px 20px 20px 5px;
    margin: 1rem 0;
    margin-right: 20%;
    box-shadow: 0 4px 15px rgba(243, 156, 18, 0.3);
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

.thinking-indicator {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}
</style>
""", unsafe_allow_html=True)

class SmartProfileAgentHybrid:
    """Agent conversationnel SmartProfile avec ChatGPT automatique pour questions hors domaine"""
    
    def __init__(self):
        self.load_data()
        self.setup_chatgpt()
        
        # Mots-clés du domaine SmartProfile marketing
        self.marketing_keywords = {
            'satisfaction', 'sentiment', 'avis', 'nps', 'feedback', 'client', 'customer',
            'campagne', 'campaign', 'email', 'sms', 'performance', 'roi', 'conversion',
            'churn', 'rétention', 'risque', 'partir', 'perdre', 'retention',
            'segment', 'profil', 'groupe', 'catégorie', 'clustering', 'segmentation',
            'recommandation', 'conseil', 'action', 'améliorer', 'optimiser', 'strategy',
            'marketing', 'digital', 'automation', 'analytics', 'data', 'crm',
            'smartprofile', 'plateforme', 'dashboard', 'kpi', 'métrique', 'revenus'
        }
    
    def load_data(self):
        """Charge des données SmartProfile"""
        np.random.seed(42)
        n_customers = 1000
        
        self.customers = pd.DataFrame({
            'customer_id': [f"SP_{i:05d}" for i in range(n_customers)],
            'sector': np.random.choice(['Banque', 'Assurance', 'Tourisme', 'Presse', 'Retail'], n_customers),
            'satisfaction': np.random.normal(3.7, 0.9, n_customers),
            'churn_risk': np.random.beta(2, 8, n_customers),
            'lifetime_value': np.random.exponential(3000, n_customers),
            'interactions': np.random.poisson(12, n_customers),
            'email_opens': np.random.poisson(6, n_customers),
            'clicks': np.random.poisson(2, n_customers),
            'channel': np.random.choice(['Email', 'SMS', 'Web', 'Mobile'], n_customers)
        })
        
        self.customers['satisfaction'] = self.customers['satisfaction'].clip(1, 5)
        self.customers['churn_risk'] = self.customers['churn_risk'].clip(0, 1)
        
        n_campaigns = 100
        self.campaigns = pd.DataFrame({
            'campaign_id': [f"CAMP_{i:04d}" for i in range(n_campaigns)],
            'channel': np.random.choice(['Email', 'SMS', 'Web', 'Mobile', 'Social'], n_campaigns),
            'roi': np.random.uniform(50, 200, n_campaigns),
            'revenue': np.random.uniform(5000, 150000, n_campaigns),
            'cost': np.random.uniform(1000, 25000, n_campaigns),
            'status': np.random.choice(['Active', 'Completed', 'Paused'], n_campaigns, p=[0.3, 0.6, 0.1])
        })
        
        n_interactions = 2000
        self.interactions = pd.DataFrame({
            'interaction_id': [f"INT_{i:06d}" for i in range(n_interactions)],
            'sentiment': np.random.choice(['positif', 'neutre', 'négatif'], n_interactions, p=[0.6, 0.25, 0.15]),
            'channel': np.random.choice(['Email', 'SMS', 'Web', 'Mobile'], n_interactions),
            'rating': np.random.randint(1, 6, n_interactions)
        })
    
    def setup_chatgpt(self):
        """Configure l'accès ChatGPT"""
        self.chatgpt_available = False
        
        # Vérifier si clé API disponible
        if 'openai_key' in st.session_state and st.session_state.openai_key:
            try:
                import openai
                openai.api_key = st.session_state.openai_key
                self.chatgpt_available = True
            except ImportError:
                st.sidebar.warning("📦 Module OpenAI non installé. pip install openai")
    
    def is_marketing_question(self, question):
        """Détermine si la question concerne le marketing SmartProfile"""
        question_lower = question.lower()
        
        # Vérifier la présence de mots-clés marketing
        marketing_score = sum(1 for keyword in self.marketing_keywords if keyword in question_lower)
        
        # Si au moins 1 mot-clé marketing trouvé, c'est du domaine
        return marketing_score > 0
    
    def ask_chatgpt_external(self, question):
        """Demande à ChatGPT pour les questions hors domaine"""
        if not self.chatgpt_available:
            return self.fallback_response(question)
        
        try:
            import openai
            
            # Prompt pour que ChatGPT réponde comme l'agent SmartProfile
            system_prompt = f"""Tu es l'assistant IA SmartProfile, un expert en marketing digital et analyse de données. 

La question suivante n'est pas directement liée au marketing, mais tu dois y répondre en tant qu'assistant SmartProfile intelligent.

Règles:
- Réponds naturellement à la question
- Reste professionnel et utile
- Si possible, fais un lien subtil avec le marketing/business
- Utilise des emojis pour la lisibilité
- Signe comme "SmartProfile AI"

Question: {question}"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return self.fallback_response(question)
    
    def fallback_response(self, question):
        """Réponse de secours si ChatGPT non disponible"""
        return f"""🤖 **SmartProfile AI - Assistant Intelligent**

Je comprends votre question : "{question}"

Bien que cette question ne soit pas directement liée à l'analyse marketing, je suis conçu principalement pour vous aider avec :

📊 **Mes spécialités marketing** :
• Analyse de satisfaction et performance client
• Optimisation des campagnes marketing multi-canal
• Prédiction et prévention du churn
• Segmentation comportementale avancée
• Recommandations ROI et allocation budget

💡 **Pour cette question spécifique**, je recommande de consulter des sources spécialisées ou d'utiliser la vraie API ChatGPT.

🔑 **Configurez l'API OpenAI** dans la sidebar pour que je puisse répondre à toutes vos questions avec l'intelligence ChatGPT !

🎯 **Puis-je vous aider avec une analyse marketing SmartProfile ?**"""
    
    def ask_smartprofile_agent(self, question):
        """Traite les questions de domaine marketing SmartProfile"""
        return self.generate_smart_response(question)
    
    def main_response_handler(self, question):
        """Gestionnaire principal qui détermine comment répondre"""
        
        # Vérifier si c'est une question marketing
        if self.is_marketing_question(question):
            # Question de domaine SmartProfile - répond avec notre expertise
            response_type = "🎯 **Expertise SmartProfile**"
            return self.ask_smartprofile_agent(question), response_type
        else:
            # Question hors domaine - demander à ChatGPT
            response_type = "🧠 **Intelligence ChatGPT via SmartProfile AI**"
            return self.ask_chatgpt_external(question), response_type
    
    def generate_smart_response(self, question):
        """Génère une réponse marketing SmartProfile"""
        q = question.lower()
        
        if any(word in q for word in ['satisfaction', 'sentiment', 'avis', 'nps']):
            return self.analyze_satisfaction()
        elif any(word in q for word in ['campagne', 'email', 'sms', 'performance', 'roi']):
            return self.analyze_campaigns()
        elif any(word in q for word in ['churn', 'rétention', 'risque', 'partir']):
            return self.analyze_churn()
        elif any(word in q for word in ['recommandation', 'conseil', 'action', 'améliorer']):
            return self.generate_recommendations()
        elif any(word in q for word in ['segment', 'profil', 'groupe', 'catégorie']):
            return self.analyze_segmentation()
        else:
            return self.marketing_help_response()
    
    def analyze_satisfaction(self):
        """Analyse de satisfaction SmartProfile"""
        avg_satisfaction = self.customers['satisfaction'].mean()
        nps = self.calculate_nps()
        high_satisfaction = (self.customers['satisfaction'] >= 4.5).sum()
        
        response = f"""📊 **Analyse Satisfaction Client SmartProfile**

😊 **Score moyen**: {avg_satisfaction:.1f}/5 
📈 **Net Promoter Score**: {nps:+.1f} {"🟢" if nps > 50 else "🟡" if nps > 0 else "🔴"}
⭐ **Clients très satisfaits**: {high_satisfaction} ({high_satisfaction/len(self.customers)*100:.1f}%)

🔍 **Détail par secteur**:
{self.get_satisfaction_by_sector()}

🎯 **Recommandations SmartProfile**:
• Programme de fidélisation pour {high_satisfaction} clients promoteurs
• Plan d'amélioration ciblé par secteur sous-performant
• Personnalisation des parcours selon niveau de satisfaction
• Intégration alerts NPS dans votre Customer Data Platform

💰 **Impact business estimé**: +15% rétention, +€{avg_satisfaction*50000:.0f} revenus annuels

🚀 **Prochaines étapes**: Mise en place scoring satisfaction temps réel"""

        # Graphique
        fig = px.histogram(self.customers, x='satisfaction', nbins=10,
                          title="Distribution Satisfaction Client SmartProfile")
        st.plotly_chart(fig, use_container_width=True)
        
        return response
    
    def analyze_campaigns(self):
        """Analyse des campagnes marketing"""
        avg_roi = self.campaigns['roi'].mean()
        best_channel = self.campaigns.groupby('channel')['roi'].mean().idxmax()
        total_revenue = self.campaigns['revenue'].sum()
        
        response = f"""📧 **Performance Campagnes SmartProfile**

📈 **ROI moyen**: {avg_roi:.1f}%
🏆 **Canal star**: {best_channel} 
💰 **Revenus générés**: €{total_revenue:,.0f}
📊 **Campagnes actives**: {len(self.campaigns[self.campaigns['status']=='Active'])}

🎯 **Top 3 canaux par ROI**:
{self.get_top_channels()}

🚀 **Optimisations recommandées**:
• Réallocation budget vers {best_channel} (+{self.campaigns[self.campaigns['channel']==best_channel]['roi'].mean()-avg_roi:.1f}% ROI vs moyenne)
• A/B testing automatisé sur créatifs sous-performants
• Orchestration cross-canal intelligente avec SmartProfile
• Personalisation par segment comportemental

💡 **Potentiel d'amélioration**: +25% ROI avec optimisation IA SmartProfile

🔮 **Vision 2024**: Campagnes auto-optimisées par IA prédictive"""

        # Graphique
        channel_perf = self.campaigns.groupby('channel')['roi'].mean()
        fig = px.bar(x=channel_perf.index, y=channel_perf.values,
                    title="ROI par Canal Marketing - SmartProfile Analytics")
        st.plotly_chart(fig, use_container_width=True)
        
        return response
    
    def analyze_churn(self):
        """Analyse du risque de churn"""
        high_risk = (self.customers['churn_risk'] > 0.7).sum()
        medium_risk = ((self.customers['churn_risk'] > 0.4) & (self.customers['churn_risk'] <= 0.7)).sum()
        avg_clv = self.customers['lifetime_value'].mean()
        
        response = f"""🔮 **Analyse Prédictive Churn - SmartProfile**

⚠️ **Clients à risque élevé**: {high_risk} ({high_risk/len(self.customers)*100:.1f}%)
🟡 **Risque modéré**: {medium_risk} ({medium_risk/len(self.customers)*100:.1f}%)
💎 **Customer Lifetime Value moyen**: €{avg_clv:.0f}
💸 **Valeur à risque immédiat**: €{high_risk * avg_clv * 0.6:.0f}

🔍 **Profiling clients à risque**:
• Secteur le plus vulnérable: {self.get_churn_by_sector()}
• Seuil engagement critique: <{self.customers['interactions'].quantile(0.25):.0f} interactions/mois
• Score satisfaction limite: <{self.customers['satisfaction'].quantile(0.3):.1f}/5

🎯 **Stratégie de rétention SmartProfile**:
• Campagne win-back automatisée pour {high_risk} clients immédiats
• Programme de fidélisation prédictif basé sur scoring
• Alerts temps réel dans votre Customer Data Platform
• Séquences de nurturing personnalisées par niveau de risque

💰 **ROI programme rétention**: €{avg_clv*0.7:.0f} par client sauvé

🚀 **Activation**: Modèle prédictif intégrable en 48h dans SmartProfile"""

        # Graphique
        fig = px.histogram(self.customers, x='churn_risk', nbins=20,
                          title="Distribution Risque de Churn - SmartProfile Predictive Analytics")
        st.plotly_chart(fig, use_container_width=True)
        
        return response
    
    def generate_recommendations(self):
        """Recommandations stratégiques SmartProfile"""
        best_channel = self.campaigns.groupby('channel')['roi'].mean().idxmax()
        worst_sector = self.customers.groupby('sector')['satisfaction'].mean().idxmin()
        
        response = f"""🎯 **Recommandations Stratégiques SmartProfile 2024**

📈 **Quick Wins - Implémentation immédiate**:
• Migration budget vers canal {best_channel} (ROI supérieur de +{self.campaigns[self.campaigns['channel']==best_channel]['roi'].mean()-self.campaigns['roi'].mean():.1f}%)
• Plan d'urgence secteur {worst_sector} (satisfaction critique détectée)
• Activation scoring comportemental temps réel sur Customer Data Platform

🚀 **Initiatives stratégiques Q2-Q3**:
• Déploiement IA prédictive multi-touch attribution
• Orchestration parcours client omnicanal automatisée
• Expansion segments high-value avec modèles look-alike
• Intégration agents conversationnels comme cette démo

🔮 **Vision SmartProfile 2024-2025**:
• Agents marketing autonomes avec prise de décision IA
• Prédiction Customer Lifetime Value en temps réel
• Personnalisation 1:1 à l'échelle avec deep learning
• Écosystème marketing automation auto-apprenant

💰 **Business Case consolidé**:
• Revenus additionnels: +€{self.campaigns['revenue'].sum()*0.3:.0f} (+30% vs baseline)
• Optimisation coûts: +25% efficacité opérationnelle
• Rétention: +40% réduction churn prévisible

🎯 **ROI programme global**: 380% sur 18 mois

📅 **Timeline activation**: Phase 1 lanceable sous 30 jours avec SmartProfile"""
        
        return response
    
    def analyze_segmentation(self):
        """Analyse de segmentation avancée"""
        response = f"""🎯 **Segmentation Comportementale SmartProfile**

📊 **Analyse automatique des {len(self.customers)} profils clients**:

🏆 **Segment Champions** ({(self.customers['satisfaction'] >= 4.5).sum()} clients):
• Satisfaction élevée + engagement fort
• CLV moyen: €{self.customers[self.customers['satisfaction'] >= 4.5]['lifetime_value'].mean():.0f}
• Stratégie: Programmes VIP et advocacy

💎 **Segment High-Value** ({((self.customers['lifetime_value'] > self.customers['lifetime_value'].quantile(0.8)) & (self.customers['churn_risk'] < 0.3)).sum()} clients):
• Forte valeur + faible risque
• Opportunités cross-sell/up-sell maximales
• Stratégie: Personnalisation premium

⚠️ **Segment At-Risk** ({((self.customers['churn_risk'] > 0.6) & (self.customers['satisfaction'] < 3.5)).sum()} clients):
• Risque élevé + satisfaction dégradée
• Valeur menacée: €{((self.customers['churn_risk'] > 0.6) & (self.customers['satisfaction'] < 3.5)).sum() * self.customers['lifetime_value'].mean() * 0.7:.0f}
• Stratégie: Win-back urgent

🔄 **Segment Nurturing** (reste de la base):
• Potentiel de montée en gamme
• Stratégie: Séquences d'engagement progressif

🚀 **Activation SmartProfile**:
• Segments mis à jour en temps réel
• Campagnes automatisées par segment
• Scoring dynamique intégré CDP
• Dashboard executif avec alertes

📈 **Impact attendu**: +45% relevance campagnes, +30% conversion"""

        # Graphique segmentation
        fig = px.scatter(self.customers, x='satisfaction', y='lifetime_value', 
                        color='churn_risk', size='interactions',
                        title="Segmentation Client SmartProfile - Vue 360°")
        st.plotly_chart(fig, use_container_width=True)
        
        return response
    
    def marketing_help_response(self):
        """Aide spécialisée marketing SmartProfile"""
        return f"""🤖 **SmartProfile Marketing Intelligence Center**

Je suis votre expert IA spécialisé en marketing digital et Customer Data Platform.

🎯 **Domaines d'expertise SmartProfile**:
• **Analytics & Performance** : ROI campagnes, attribution multi-touch, KPIs avancés
• **Customer Intelligence** : Segmentation comportementale, scoring prédictif, CLV
• **Marketing Automation** : Orchestration omnicanale, personnalisation, nurturing
• **Prédictif & IA** : Churn prevention, recommandations, modèles prédictifs

💡 **Questions marketing que je maîtrise**:
• "Analyse la performance de nos campagnes email vs SMS par secteur"
• "Identifie les 100 clients à plus fort potentiel de cross-sell"
• "Optimise l'allocation de notre budget Q2 par canal et segment"
• "Prédis l'impact d'une campagne de rétention sur le secteur Banque"

📊 **Ma base de données actuelle**:
• {len(self.customers):,} profils clients analysés
• {len(self.campaigns)} campagnes avec métriques complètes  
• {len(self.interactions):,} interactions comportementales

🔑 **Mode hybride activé** : Questions marketing = expertise SmartProfile | Questions générales = Intelligence ChatGPT

🎯 **Comment puis-je optimiser votre stratégie marketing aujourd'hui ?**"""
    
    # Méthodes utilitaires
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

def main():
    """Interface principale"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 SmartProfile Marketing AI</h1>
        <p>Assistant Conversationnel Hybride - Expertise Marketing + Intelligence ChatGPT</p>
        <small>Customer Data Platform • Modèles Prédictifs • IA Généraliste</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation de l'agent
    if 'agent' not in st.session_state:
        with st.spinner("🚀 Initialisation SmartProfile AI Hybride..."):
            st.session_state.agent = SmartProfileAgentHybrid()
            st.session_state.messages = []
        st.success("✅ Agent SmartProfile Hybride activé !")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="status-badge">SmartProfile AI Hybride</div>', unsafe_allow_html=True)
        
        # Status API
        if st.session_state.agent.chatgpt_available:
            st.markdown('<div class="api-status">🧠 ChatGPT Connecté</div>', unsafe_allow_html=True)
            st.success("🔥 Mode Hybride Complet Activé")
        else:
            st.markdown('<div class="api-status">🎯 Mode Marketing + Simulation</div>', unsafe_allow_html=True)
            
            # Configuration API
            with st.expander("🔑 Configuration ChatGPT"):
                st.info("Pour les questions hors marketing, activez ChatGPT")
                api_key = st.text_input("Clé API OpenAI:", type="password", 
                                       help="Questions marketing = expertise SmartProfile\nQuestions générales = ChatGPT")
                if api_key:
                    st.session_state.openai_key = api_key
                    st.session_state.agent.chatgpt_available = True
                    st.success("✅ ChatGPT activé pour questions générales!")
                    st.rerun()
        
        # Métriques live
        st.subheader("📊 Analytics SmartProfile")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👥 Clients", f"{len(st.session_state.agent.customers):,}")
            st.metric("📧 Campagnes", len(st.session_state.agent.campaigns))
        with col2:
            avg_satisfaction = st.session_state.agent.customers['satisfaction'].mean()
            st.metric("😊 NPS", f"{st.session_state.agent.calculate_nps():+.0f}")
            avg_roi = st.session_state.agent.campaigns['roi'].mean()
            st.metric("📈 ROI", f"{avg_roi:.1f}%")
        
        # Questions prédéfinies mixtes
        st.subheader("💡 Questions Exemples")
        
        st.write("🎯 **Questions Marketing** (expertise SmartProfile):")
        marketing_questions = [
            "Analyse la satisfaction client par secteur",
            "Performance des campagnes email vs SMS", 
            "Identifie les clients à risque de churn",
            "Recommande des actions marketing",
            "Optimise la segmentation comportementale"
        ]
        
        for question in marketing_questions:
            if st.button(question, key=f"mkt_{hash(question)}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("🎯 Analyse expertise SmartProfile..."):
                    response, response_type = st.session_state.agent.main_response_handler(question)
                st.session_state.messages.append({"role": "assistant", "content": response, "type": response_type})
                st.rerun()
        
        st.write("🧠 **Questions Générales** (via ChatGPT):")
        general_questions = [
            "Explique-moi la théorie de la relativité",
            "Recette pour faire un gâteau au chocolat",
            "Histoire de la Révolution française",
            "Comment apprendre le piano ?",
        ]
        
        for question in general_questions:
            if st.button(question, key=f"gen_{hash(question)}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("🧠 Consultation ChatGPT..."):
                    response, response_type = st.session_state.agent.main_response_handler(question)
                st.session_state.messages.append({"role": "assistant", "content": response, "type": response_type})
                st.rerun()
    
    # Dashboard métriques principales
    st.subheader("📊 Dashboard SmartProfile Live")
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
    st.subheader("💬 Chat avec SmartProfile AI Hybride")
    
    # Affichage des messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-user"><strong>👤 Vous :</strong> {message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            # Afficher le type de réponse si disponible
            response_type = message.get("type", "")
            if response_type:
                st.markdown(f'<small style="color: #666; margin-left: 20%;">{response_type}</small>', 
                           unsafe_allow_html=True)
            
            st.markdown(f'<div class="chat-assistant"><strong>🤖 SmartProfile AI :</strong><br>{message["content"]}</div>', 
                       unsafe_allow_html=True)
    
    # Input utilisateur
    user_input = st.chat_input("Posez votre question (marketing ou générale)...")
    
    if user_input:
        # Ajouter message utilisateur
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Déterminer le type de question et afficher l'indicateur approprié
        is_marketing = st.session_state.agent.is_marketing_question(user_input)
        
        if is_marketing:
            status_text = "🎯 Analyse avec expertise SmartProfile marketing..."
        else:
            status_text = "🧠 Consultation de ChatGPT pour question générale..."
        
        # Générer réponse avec indicateur de type
        with st.spinner(status_text):
            time.sleep(1)
            response, response_type = st.session_state.agent.main_response_handler(user_input)
        
        # Ajouter réponse avec type
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "type": response_type
        })
        st.rerun()

if __name__ == "__main__":
    main()