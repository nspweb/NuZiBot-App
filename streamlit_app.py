import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from chatbot_model import NutritionChatbot
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="🍎 NuZiBot - Nutrition Education Chatbot",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme management
theme = st.sidebar.radio("🌙 Theme", ["Light", "Dark"], index=0)

# Fixed CSS template with properly escaped braces
def get_theme_css(theme_colors):
    return f"""
<style>
    .main, .sidebar .sidebar-content {{
        background-color: {theme_colors['background_color']};
        color: {theme_colors['text_color']};
    }}
    .main-header, .chat-container, .metric-card, .footer, .dashboard-card {{
        background-color: {theme_colors['card_bg_color']};
        color: {theme_colors['card_text_color']};
        border: 1px solid {theme_colors['border_color']};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }}
    .user-message {{
        background-color: {theme_colors['user_bg_color']};
        color: {theme_colors['user_text_color']};
        border-left: 5px solid {theme_colors['user_border_color']};
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }}
    .bot-message {{
        background-color: {theme_colors['bot_bg_color']};
        color: {theme_colors['bot_text_color']};
        border-left: 5px solid {theme_colors['bot_border_color']};
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }}
    .bot-message-warning {{
        background-color: {theme_colors['warning_bg_color']};
        color: {theme_colors['warning_text_color']};
        border-left: 5px solid {theme_colors['warning_border_color']};
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }}
    .warning-banner {{
        background-color: {theme_colors['warning_bg_color']};
        color: {theme_colors['warning_text_color']};
        border-left: 5px solid {theme_colors['warning_border_color']};
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }}
    .info-banner {{
        background-color: {theme_colors['user_bg_color']};
        color: {theme_colors['user_text_color']};
        border-left: 5px solid {theme_colors['user_border_color']};
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }}
    .metric-card {{
        text-align: center;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
    }}
    .sidebar .stRadio > label {{
        color: {theme_colors['text_color']};
    }}
    .dashboard-card {{
        border: 1px solid {theme_colors['border_color']};
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
</style>
"""

# Apply theme
if theme == "Light":
    theme_colors = {
        'background_color': "#ffffff",
        'text_color': "#000000", 
        'card_bg_color': "#e0f2e9",
        'card_text_color': "#2e7d32",
        'border_color': "#d3e0d4",
        'user_bg_color': "#e3f2fd",
        'user_text_color': "#1565c0",
        'user_border_color': "#2196f3",
        'bot_bg_color': "#e8f5e9",
        'bot_text_color': "#2e7d32",
        'bot_border_color': "#4caf50",
        'warning_bg_color': "#fff3e0",
        'warning_text_color': "#ef6c00",
        'warning_border_color': "#ff9800"
    }
else:  # Dark
    theme_colors = {
        'background_color': "#1e1e1e",
        'text_color': "#ffffff",
        'card_bg_color': "#2e2e2e",
        'card_text_color': "#ffffff",
        'border_color': "#404040",
        'user_bg_color': "#1a237e",
        'user_text_color': "#ffffff",
        'user_border_color': "#3f51b5",
        'bot_bg_color': "#1b5e20",
        'bot_text_color': "#ffffff",
        'bot_border_color': "#4caf50",
        'warning_bg_color': "#e65100",
        'warning_text_color': "#ffffff",
        'warning_border_color': "#ff9800"
    }

st.markdown(get_theme_css(theme_colors), unsafe_allow_html=True)

# Initialize session state
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = "groq"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

# Load chatbot
@st.cache_resource
def load_chatbot(llm_type):
    return NutritionChatbot(
        food_csv_path="merged_food_with_ingredients.csv",
        nutrition_excel_path="Recommended Dietary Allowances and Adequate Intakes Total Water and Macronutrients.xlsx",
        llm_type=llm_type
    )

chatbot = load_chatbot(st.session_state.selected_llm)

# Header
st.markdown("""
<div class="main-header">
    <h1>🍎 NuZiBot</h1>
    <h3>Intelligent Nutrition Assistant for Children & Adolescents</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar with navigation and instructions
with st.sidebar:
    st.markdown("### 📋 Navigation")
    
    # Check if profile is complete
    profile_complete = bool(st.session_state.user_profile)
    
    # Show instructions
    if not profile_complete:
        st.markdown("""
        <div class="info-banner">
            <h4>📝 Getting Started:</h4>
            <p><strong>Step 1:</strong> Complete your profile in Dashboard</p>
            <p><strong>Step 2:</strong> Start chatting for personalized nutrition advice</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-banner">
            <h4>✅ Profile Complete!</h4>
            <p>You can now chat with NuZiBot for personalized nutrition advice.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation buttons
    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.current_page = "Dashboard"
    
    if profile_complete:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.current_page = "Chat"
    else:
        st.button("💬 Chat (Complete Profile First)", use_container_width=True, disabled=True)
    
    # Profile status
    st.markdown("---")
    st.markdown("### 👤 Profile Status")
    if profile_complete:
        st.success("✅ Profile Complete")
        profile = st.session_state.user_profile
        st.write(f"**Age:** {profile.get('age', 'N/A')} years")
        st.write(f"**Gender:** {profile.get('gender', 'N/A')}")
        st.write(f"**BMI:** {profile.get('bmi', 0):.1f}")
    else:
        st.warning("⚠️ Profile Incomplete")
        st.write("Please complete your profile in the Dashboard.")

# Set default page
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Dashboard Page
if st.session_state.current_page == "Dashboard":
    st.markdown("""
    <div class="chat-container">
        <h2>📊 User Profile & Nutrition Dashboard</h2>
        <p>Complete your profile to get personalized nutrition recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("user_profile_form"):
        st.subheader("👤 Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=st.session_state.user_profile.get('weight', 50.0))
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=st.session_state.user_profile.get('height', 160.0))
            age = st.number_input("Age (years)", min_value=1, max_value=100, value=st.session_state.user_profile.get('age', 15))
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.user_profile.get('gender') == 'Male' else 1)
            activity_level = st.selectbox("Activity Level", 
                ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"],
                index=["Sedentary", "Lightly Active", "Moderately Active", "Very Active"].index(st.session_state.user_profile.get('activity_level', 'Moderately Active')))
        
        st.subheader("🏥 Health & Dietary Information")
        special_conditions = st.multiselect("Special Health Conditions", 
            ["Diabetes", "Hypertension", "Nut Allergy", "Milk Allergy", "Vegetarian", "Vegan", "Lactose Intolerant"],
            default=st.session_state.user_profile.get('special_conditions', []))
        dietary_preferences = st.text_area("Dietary Preferences / Food Restrictions", 
            value=st.session_state.user_profile.get('dietary_preferences', ''),
            placeholder="e.g., No spicy foods, prefers fruits, dislikes vegetables...")
        
        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
        
        if submitted:
            # Validation warnings
            warnings = []
            if "Vegan" in special_conditions and any(cond in ["Milk Allergy", "Lactose Intolerant"] for cond in special_conditions):
                warnings.append("⚠️ Vegan diet already excludes dairy products.")
            if dietary_preferences and "vegan" in dietary_preferences.lower() and "Vegan" not in special_conditions:
                warnings.append("⚠️ Consider selecting 'Vegan' in Special Conditions if you follow a vegan diet.")
            
            for warning in warnings:
                st.markdown(f'<div class="warning-banner">{warning}</div>', unsafe_allow_html=True)
            
            # Calculate BMI, BMR, and energy needs
            height_m = height / 100
            bmi = weight / (height_m ** 2)
            
            # Mifflin-St Jeor Equation for BMR
            if gender == "Male":
                bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
            else:
                bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
            
            # Activity multipliers
            activity_multipliers = {
                "Sedentary": 1.2,
                "Lightly Active": 1.375,
                "Moderately Active": 1.55,
                "Very Active": 1.725
            }
            target_energy = bmr * activity_multipliers[activity_level]
            
            # Save profile
            st.session_state.user_profile = {
                "weight": weight,
                "height": height,
                "age": age,
                "gender": gender,
                "bmi": bmi,
                "bmr": bmr,
                "target_energy_intake": target_energy,
                "activity_level": activity_level,
                "special_conditions": special_conditions,
                "dietary_preferences": dietary_preferences
            }
            
            chatbot.update_user_profile(st.session_state.user_profile)
            st.success("✅ Profile saved successfully! You can now use the Chat feature.")
            st.balloons()
    
    # Display dashboard if profile exists
    if st.session_state.user_profile:
        profile = st.session_state.user_profile
        nutrition_needs = chatbot.get_nutrition_recommendations(profile)
        
        st.markdown("---")
        st.subheader("📈 Your Nutrition Dashboard")
        
        # BMI Visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 📏 BMI Status")
            
            # BMI categories
            if profile['bmi'] < 18.5:
                bmi_status = "Underweight"
                bmi_color = "#2196f3"
            elif 18.5 <= profile['bmi'] < 25:
                bmi_status = "Normal"
                bmi_color = "#4caf50"
            elif 25 <= profile['bmi'] < 30:
                bmi_status = "Overweight"
                bmi_color = "#ff9800"
            else:
                bmi_status = "Obese"
                bmi_color = "#f44336"
            
            fig_bmi = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=profile['bmi'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"BMI: {bmi_status}"},
                gauge={
                    'axis': {'range': [0, 40]},
                    'bar': {'color': bmi_color},
                    'steps': [
                        {'range': [0, 18.5], 'color': "lightblue"},
                        {'range': [18.5, 25], 'color': "lightgreen"},
                        {'range': [25, 30], 'color': "orange"},
                        {'range': [30, 40], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 25
                    }
                }
            ))
            fig_bmi.update_layout(height=300)
            st.plotly_chart(fig_bmi, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 🍽️ Daily Macronutrient Goals")
            
            macro_data = {
                "Macronutrient": ["Carbohydrates", "Protein", "Fat"],
                "Grams": [
                    nutrition_needs.get("Carbohydrate (g/d)", 0),
                    nutrition_needs.get("Protein (g/d)", 0),
                    nutrition_needs.get("Fat (g/d)", 0)
                ]
            }
            
            fig_pie = px.pie(
                names=macro_data["Macronutrient"],
                values=macro_data["Grams"],
                color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99']
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Nutrition metrics cards
        st.markdown("### 📊 Daily Nutrition Targets")
        cols = st.columns(5)
        
        metrics = [
            ("🍞", "Carbs", nutrition_needs.get("Carbohydrate (g/d)", 0), "g", "#ff9999"),
            ("🥩", "Protein", nutrition_needs.get("Protein (g/d)", 0), "g", "#66b3ff"),
            ("🥑", "Fat", nutrition_needs.get("Fat (g/d)", 0), "g", "#99ff99"),
            ("🌾", "Fiber", nutrition_needs.get("Total Fiber (g/d)", 0), "g", "#ffcc99"),
            ("💧", "Water", nutrition_needs.get("Total Water (L/d)", 0), "L", "#87ceeb")
        ]
        
        for i, (icon, name, value, unit, color) in enumerate(metrics):
            with cols[i]:
                st.markdown(f'''
                <div class="metric-card" style="background-color: {color};">
                    <h2>{icon}</h2>
                    <h3>{value:.1f}{unit}</h3>
                    <p><strong>{name}</strong></p>
                </div>
                ''', unsafe_allow_html=True)

# Chat Page
elif st.session_state.current_page == "Chat":
    if not st.session_state.user_profile:
        st.markdown("""
        <div class="warning-banner">
            <h3>⚠️ Profile Required</h3>
            <p>Please complete your profile in the Dashboard first to get personalized nutrition advice.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📊 Go to Dashboard", use_container_width=True):
            st.session_state.current_page = "Dashboard"
            st.rerun()
    else:
        st.markdown("""
        <div class="chat-container">
            <h2>💬 Chat with NuZiBot</h2>
            <p>Ask me about healthy meals, nutrition tips, or dietary recommendations!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        for user_msg, bot_msg in st.session_state.chat_history:
            st.markdown(f"""
            <div class="user-message">
                <strong>👤 You:</strong> {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            if "Sorry, that question is outside the topic of nutrition" in bot_msg:
                st.markdown(f"""
                <div class="bot-message-warning">
                    <strong>🤖 NuZiBot:</strong> {bot_msg}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 NuZiBot:</strong> {bot_msg}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("💭 Ask about healthy meals, nutrition tips, or dietary needs...")
        if user_input:
            with st.spinner("🤔 NuZiBot is thinking..."):
                try:
                    # SOLUSI 1: Hapus parameter context jika method tidak mendukungnya
                    response = chatbot.get_response(user_input)
                    
                    # SOLUSI 2: Atau jika ingin mengirim profil user, gabungkan dengan input
                    # user_input_with_context = f"User profile: {json.dumps(st.session_state.user_profile)}\n\nQuestion: {user_input}"
                    # response = chatbot.get_response(user_input_with_context)
                    
                    st.session_state.chat_history.append((user_input, response))
                    st.rerun()
                except Exception as e:
                    st.error(f"Sorry, there was an error: {str(e)}")

# Action buttons
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()

with col_btn2:
    if st.button("🔄 Reset Profile", use_container_width=True):
        st.session_state.user_profile = {}
        st.session_state.current_page = "Dashboard"
        st.success("Profile reset! Please complete your profile again.")
        st.rerun()

with col_btn3:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = "Dashboard"
        st.rerun()