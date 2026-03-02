import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from fpdf import FPDF
from io import BytesIO
from PIL import Image
import io
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =============================
# PROFESSIONAL UI/UX STYLING
# =============================
st.set_page_config(page_title="TravelAI | Elite Cultural Tourism Platform", page_icon="✈️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1e293b; }
    .main { background-color: #f8fafc; }
    .main-title { font-size: 3rem; font-weight: 800; letter-spacing: -1px; color: #0f172a; text-align: left; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.1rem; color: #64748b; margin-bottom: 3rem; }
    .input-card { background: #ffffff; padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem; }
    .stButton > button { background: #2563eb !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; padding: 0.75rem 2rem !important; width: 100% !important; height: 55px !important; }
    .stButton > button:hover { background: #1d4ed8 !important; box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4) !important; transform: translateY(-2px); }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; color: #2563eb !important; }
    .map-container { border-radius: 20px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05); }
    
    /* Sidebar Styling for Hodophiler */
    section[data-testid="sidebar"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; }
    .hodophiler-header { background: rgba(255,255,255,0.1); backdrop-filter: blur(20px); border-radius: 20px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center; }
    .chat-bubble-user { background: rgba(255,255,255,0.2); border-radius: 18px; padding: 12px 16px; margin: 8px 0; color: white; backdrop-filter: blur(10px); }
    .chat-bubble-assistant { background: rgba(255,255,255,0.15); border-radius: 18px; padding: 12px 16px; margin: 8px 0; color: white; backdrop-filter: blur(10px); }
</style>
""", unsafe_allow_html=True)

# =============================
# SETUP - OpenRouter API
# =============================
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

@st.cache_data
def load_data():
    try:
        path = "master_dataset_week2.csv"
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("(", "").str.replace(")", "").str.replace("/", "_")
        return df
    except: 
        st.error("❌ CSV file not found!")
        return pd.DataFrame()

df = load_data()

LANGUAGE_MAP = {
    "English": "English", "Spanish": "Spanish", "French": "French", "Hindi": "Hindi", "German": "German"
}

# =============================
# FEEDBACK STORAGE FUNCTIONS (NEW)
# =============================
FEEDBACK_FILE = "feedback_data.json"

def load_feedback():
    """Load all feedback from JSON file."""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []

def save_feedback(entry):
    """Append a new feedback entry and save."""
    data = load_feedback()
    data.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_feedback_df():
    """Return feedback as a cleaned DataFrame."""
    data = load_feedback()
    if not data:
        return pd.DataFrame()
    df_fb = pd.DataFrame(data)
    df_fb["timestamp"] = pd.to_datetime(df_fb["timestamp"])
    return df_fb

def analyze_feedback_trends(df_fb):
    """Compute key insight metrics from feedback."""
    if df_fb.empty:
        return {}
    insights = {
        "total_responses": len(df_fb),
        "avg_itinerary_rating": round(df_fb["itinerary_rating"].mean(), 2),
        "avg_recommendation_rating": round(df_fb["recommendation_rating"].mean(), 2),
        "avg_video_rating": round(df_fb["video_rating"].mean(), 2),
        "avg_chatbot_rating": round(df_fb["chatbot_rating"].mean(), 2),
        "top_destination": df_fb["destination"].value_counts().idxmax() if "destination" in df_fb.columns else "N/A",
        "top_language": df_fb["language"].value_counts().idxmax() if "language" in df_fb.columns else "N/A",
        "top_season": df_fb["season"].value_counts().idxmax() if "season" in df_fb.columns else "N/A",
    }
    return insights

def adaptive_boost(row, df_fb):
    """
    Adaptive improvement: boost score for destinations/languages
    that have high average feedback ratings (>= 4.0).
    """
    if df_fb.empty:
        return 0
    boost = 0
    dest_fb = df_fb[df_fb["destination"] == row.get("city", "")]
    if not dest_fb.empty and dest_fb["itinerary_rating"].mean() >= 4.0:
        boost += 15  # Strong boost for well-rated destinations
    elif not dest_fb.empty and dest_fb["itinerary_rating"].mean() >= 3.0:
        boost += 5   # Mild boost
    return boost

# =============================
# CORE FUNCTIONS
# =============================
def age_group(age): 
    return "Young Explorer" if age < 25 else "Balanced Traveler" if age < 50 else "Leisure Traveler"

def find_best_destination(home_country, travel_country, budget, min_rating, interests, age, duration_days):
    home_df = df[df['country'] == home_country]
    travel_df = df[df['country'] == travel_country]
    
    if home_df.empty or travel_df.empty: 
        return None
    
    # Load feedback for adaptive scoring
    df_fb = get_feedback_df()

    def ultimate_score(row):
        interest_score = sum(row.get(interest.lower(), 0) for interest in interests)
        budget_score = 20 if row["budget_level"] == budget else 10
        rating_score = row["avg_rating"] * 8
        home_bonus = 15 if not home_df[(home_df['budget_level']==budget)&(home_df['avg_rating']>=min_rating)].empty else 0
        duration_bonus = 10 if duration_days >= 5 else 5
        # ADAPTIVE BOOST from feedback trends
        feedback_boost = adaptive_boost(row, df_fb)
        return interest_score + budget_score + rating_score + home_bonus + duration_bonus + feedback_boost
    
    scored = travel_df[travel_df['avg_rating'] >= min_rating].copy()
    if scored.empty: 
        return None
    
    scored['ultimate_score'] = scored.apply(ultimate_score, axis=1)
    return scored.loc[scored['ultimate_score'].idxmax()]

def generate_consolidated_itinerary(row, profile, duration, language, interests, home_country):
    days = "2" if "2" in duration else "5" if "5" in duration else "7"
    
    prompt = f"""
    ELITE CONSOLIDATED ITINERARY in {language}:
    Home: {home_country} → {row['destination_name']}, {row['country']}
    
    Create DETAILED {days}-day itinerary in {language}:
    ## {row['city']} MASTER PLAN ({days} Days)
    ### Day 1: [THEME]
    ### Day 2: [THEME]  
    ### Day 3-{days}: [THEME]
    ### PRO TIPS
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini", 
        messages=[{"role":"user","content":prompt}], 
        temperature=0.8
    )
    return response.choices[0].message.content

def create_pdf(text, row, rating, duration, language="English"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 22)
    pdf.cell(0, 15, f"TravelAI Elite Guide ({language})", ln=True, align='C')
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, f"{row['city']}, {row['country']}", ln=True, align='C')
    pdf.ln(8)

    pdf.set_font("Arial", "", 12)
    details = [
        f"Language: {language}",
        f"From: {row['country']}",
        f"Climate: {row['climate_label']}",
        f"Daily Cost: ${row['avg_cost_usd_day']}",
        f"Rating: {row['avg_rating']}",
        f"Duration: {duration}"
    ]
    for detail in details:
        pdf.cell(0, 8, detail, ln=True)
    pdf.ln(12)

    safe_text = str(text).encode('latin-1', 'replace').decode('latin-1')
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, safe_text)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 12, f"Your Rating: {rating}/5 Stars", ln=True, align='C')

    # ✅ FIXED: write to temp file then read back as clean bytes
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
    pdf.output(tmp_path)
    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()
    os.remove(tmp_path)

    buffer = BytesIO(pdf_bytes)
    buffer.seek(0)
    return buffer
def generate_cinematic_gif(spot, language, duration):
    import numpy as np

    frames = []
    city = spot.get('city', 'Your Destination')
    country = spot.get('country', '')
    rating = spot.get('avg_rating', 4.5)
    cost = spot.get('avg_cost_usd_day', 0)
    temp = spot.get('avg_temp_monthly', 25)
    climate = spot.get('climate_label', 'Pleasant')
    budget = spot.get('budget_level', 'Mid-Range')
    days = "2" if "2" in duration else "5" if "5" in duration else "7"

    # ── SLIDE DEFINITIONS ──────────────────────────────────────────────────────
    # Each entry: (bg_gradient_colors, headline, subline, detail_lines, accent_color)
    slides = [
        # 1 — Opening title
        (['#0f0c29', '#302b63', '#24243e'],
         f"✈️  WELCOME TO",
         f"{city.upper()}",
         [f"📍 {country}  •  🌐 {language}", f"Your Elite {days}-Day Journey Awaits"],
         '#FFD700'),

        # 2 — Destination snapshot
        (['#1a1a2e', '#16213e', '#0f3460'],
         "🌍  DESTINATION PROFILE",
         f"{city}, {country}",
         [f"⭐  Rating:  {rating:.1f} / 5.0",
          f"💰  Daily Cost:  ${cost:.0f} USD",
          f"🌡️  Avg Temp:  {temp}°C  •  {climate}",
          f"🏷️  Budget Level:  {budget}"],
         '#4ECDC4'),

        # 3 — Day 1
        (['#134e5e', '#71b280'],
         f"📅  DAY 1  OF  {days}",
         "Arrival & First Impressions",
         ["🛬  Land, check-in & freshen up",
          "🏙️  Evening stroll through the old town",
          "🍽️  Welcome dinner — taste local cuisine",
          "🌃  Soak in the city lights"],
         '#FFD700'),

        # 4 — Day 2
        (['#833ab4', '#fd1d1d', '#fcb045'],
         f"📅  DAY 2  OF  {days}",
         "Culture & Heritage Deep-Dive",
         ["🏛️  Morning: iconic museums & monuments",
          "🎨  Afternoon: art districts & galleries",
          "🛍️  Local markets & handcrafted souvenirs",
          "🎭  Evening: cultural show or live music"],
         '#FF6B6B'),

        # 5 — Day 3 (shown for 5/7-day trips as well)
        (['#0f2027', '#203a43', '#2c5364'],
         f"📅  DAY 3  OF  {days}",
         "Adventure & Nature Escapes",
         ["🌿  Day trip to nature reserves / hills",
          "🚴  Cycling or trekking trails",
          "📸  Golden-hour photography spots",
          "🧘  Sunset yoga or wellness retreat"],
         '#96CEB4'),

        # 6 — Extended days (5 / 7)
        (['#232526', '#414345'],
         f"📅  DAYS 4–{days}",
         "Hidden Gems & Local Life",
         ["☕  Neighbourhood café mornings",
          "🤝  Interact with local communities",
          "🍲  Street-food crawl & cooking class",
          "🏖️  Leisure beach / spa / free exploration"],
         '#45B7D1'),

        # 7 — Travel stats / value
        (['#1d1d1d', '#3a3a3a'],
         "💎  YOUR TRIP AT A GLANCE",
         f"{days}-Day {city} Experience",
         [f"📆  Duration:  {duration}",
          f"💵  Est. Total:  ${float(cost)*int(days):.0f} USD",
          f"⭐  Destination Rating:  {rating:.1f} / 5",
          f"🌐  Guide Language:  {language}"],
         '#FECA57'),

        # 8 — Pro tips
        (['#0d324d', '#7f5a83'],
         "💡  PRO TRAVELER TIPS",
         f"Make the Most of {city}",
         ["🕐  Arrive at attractions before 9 AM",
          "💳  Carry local currency + a travel card",
          "📱  Download offline maps in advance",
          "🤫  Ask locals for secret dining spots"],
         '#FF6B6B'),

        # 9 — Safety & essentials
        (['#1a1a1a', '#2d2d2d'],
         "🛡️  TRAVEL ESSENTIALS",
         "Stay Safe & Prepared",
         ["🏥  Keep emergency contacts saved",
          "🌂  Check weather before each day",
          "🔋  Portable charger is a must",
          "🧳  Pack light — one carry-on is ideal"],
         '#4ECDC4'),

        # 10 — Closing credits
        (['#0f0c29', '#302b63', '#24243e'],
         "🌟  BON VOYAGE!",
         f"Your {city} Story Begins Now",
         [f"✈️  Safe travels from TravelAI Premium",
          f"🗺️  Powered by Hodophiler AI",
          f"⭐  Rated {rating:.1f}/5 by fellow travelers",
          "💙  Share your memories with #TravelAI"],
         '#FFD700'),
    ]

    # ── FRAME RENDERER ─────────────────────────────────────────────────────────
    for idx, (bg_colors, headline, title, details, accent) in enumerate(slides):
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.patch.set_facecolor(bg_colors[0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # ── background gradient (vertical bands) ──
        n_bands = 300
        for k in range(n_bands):
            t = k / n_bands
            if len(bg_colors) == 2:
                c1 = plt.matplotlib.colors.to_rgb(bg_colors[0])
                c2 = plt.matplotlib.colors.to_rgb(bg_colors[1])
                col = tuple(c1[j] * (1 - t) + c2[j] * t for j in range(3))
            else:
                c1 = plt.matplotlib.colors.to_rgb(bg_colors[0])
                c2 = plt.matplotlib.colors.to_rgb(bg_colors[1])
                c3 = plt.matplotlib.colors.to_rgb(bg_colors[2])
                if t < 0.5:
                    col = tuple(c1[j] * (1 - 2*t) + c2[j] * 2*t for j in range(3))
                else:
                    col = tuple(c2[j] * (2 - 2*t) + c3[j] * (2*t - 1) for j in range(3))
            ax.axhspan(k / n_bands, (k + 1) / n_bands, color=col, linewidth=0)

        # ── decorative top accent bar ──
        ax.axhspan(0.93, 1.0, color=accent, alpha=0.85, linewidth=0)

        # ── decorative bottom bar ──
        ax.axhspan(0.0, 0.055, color=accent, alpha=0.25, linewidth=0)

        # ── slide number dots ──
        total = len(slides)
        dot_spacing = 0.028
        start_x = 0.5 - (total - 1) * dot_spacing / 2
        for d in range(total):
            dot_color = accent if d == idx else '#555555'
            dot_size  = 120 if d == idx else 60
            ax.scatter(start_x + d * dot_spacing, 0.028, s=dot_size,
                       color=dot_color, zorder=5)

        # ── headline (small label) ──
        ax.text(0.5, 0.88, headline,
                ha='center', va='center', fontsize=18,
                color=accent, weight='bold',
                transform=ax.transAxes)

        # ── main title ──
        ax.text(0.5, 0.76, title,
                ha='center', va='center', fontsize=42, weight='black',
                color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='none',
                          edgecolor=accent, linewidth=2, alpha=0.6),
                transform=ax.transAxes)

        # ── divider line ──
        ax.axhline(y=0.665, xmin=0.1, xmax=0.9, color=accent, linewidth=1.5, alpha=0.7)

        # ── detail lines ──
        line_positions = [0.58, 0.505, 0.43, 0.355]
        for li, (line_text, y_pos) in enumerate(zip(details, line_positions)):
            ax.text(0.5, y_pos, line_text,
                    ha='center', va='center', fontsize=19,
                    color='white' if li % 2 == 0 else '#d0d0d0',
                    transform=ax.transAxes)

        # ── watermark / branding ──
        ax.text(0.5, 0.085, f"TravelAI Premium  •  {city}, {country}  •  {language}",
                ha='center', va='center', fontsize=13,
                color='rgba(255,255,255,0.55)' if False else '#aaaaaa',
                transform=ax.transAxes)

        # ── corner logo mark ──
        ax.text(0.97, 0.965, "✈️", ha='right', va='center', fontsize=22,
                color='white', transform=ax.transAxes)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        img = Image.open(buf).copy()
        frames.append(img)
        buf.close()
        plt.close(fig)

    # ── BUILD GIF ──────────────────────────────────────────────────────────────
    # Vary durations: title/closing hold longer, middle slides normal pace
    durations = []
    for i in range(len(frames)):
        if i in (0, len(frames) - 1):
            durations.append(1800)   # 1.8 s — title & closing
        elif i in (1, 6):
            durations.append(1400)   # 1.4 s — info slides
        else:
            durations.append(1100)   # 1.1 s — day slides

    gif_buffer = BytesIO()
    frames[0].save(
        gif_buffer, format='GIF',
        save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True
    )
    gif_buffer.seek(0)
    return gif_buffer

# =============================
# HODOPHILER SIDEBAR CHATBOT (FIXED)
# =============================
with st.sidebar:
    st.markdown("""
    <div class="hodophiler-header">
        <h2 style='color: white; font-size: 2.2rem; margin: 0; font-weight: 800;'>🗺️ Hodophiler</h2>
        <p style='color: rgba(255,255,255,0.9); font-size: 1rem; margin: 0;'>Your Travel Whisperer AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize sidebar chat messages
    if "sidebar_messages" not in st.session_state:
        st.session_state.sidebar_messages = []
    
    # Display chat history
    chat_container = st.container(height=150)
    with chat_container:
        for message in st.session_state.sidebar_messages[-8:]:  # Last 8 messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Get trip context if available
    trip_context = {}
    if "best_spot" in st.session_state:
        spot = st.session_state.best_spot
        trip_context = {
            "city": spot.get('city', 'Unknown'),
            "country": spot.get('country', 'Unknown'),
            "interests": st.session_state.get('interests', ['Culture']),
            "temp": spot.get('avg_temp_monthly', 'Pleasant'),
            "climate": spot.get('climate_label', 'Moderate')
        }
    
    # Chat input
    if prompt := st.chat_input("Ask about your trip...", key="sidebar_input"):
        # Add user message
        st.session_state.sidebar_messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with chat_container:
            with st.chat_message("assistant"):
                if trip_context:
                    chat_prompt = f"""
                    Hodophiler - TravelAI Assistant for {trip_context['city']}, {trip_context['country']}.
                    User interests: {trip_context['interests']}
                    Weather: {trip_context['temp']}°C, {trip_context['climate']}
                    
                    Question: "{prompt}"
                    Answer helpfully with local tips!
                    """
                else:
                    chat_prompt = f"Hodophiler - Travel expert. Question: '{prompt}'"
                
                with st.spinner('Hodophiler thinking...'):
                    try:
                        response = client.chat.completions.create(
                            model="openai/gpt-4o-mini",
                            messages=[{"role": "user", "content": chat_prompt}],
                            temperature=0.8
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                        st.session_state.sidebar_messages.append({"role": "assistant", "content": answer})
                    except:
                        st.error("Chat temporarily unavailable")
        
        st.rerun()
    
    # Quick buttons (using unique keys)
    if st.button("❓ What to pack?", key="quick_pack"):
        question = "What should I pack?"
        if "best_spot" in st.session_state:
            question = f"What should I pack for {st.session_state.best_spot.get('city', '')}?"
        st.session_state.sidebar_messages.append({"role": "user", "content": question})
        st.rerun()
    
    if st.button("🍴 Best food?", key="quick_food"):
        question = "Best restaurants?"
        if "best_spot" in st.session_state:
            question = f"Best restaurants in {st.session_state.best_spot.get('city', '')}?"
        st.session_state.sidebar_messages.append({"role": "user", "content": question})
        st.rerun()
    
    if st.button("🚌 Transport?", key="quick_transport"):
        question = "How to get around?"
        if "best_spot" in st.session_state:
            question = f"How to get around in {st.session_state.best_spot.get('city', '')}?"
        st.session_state.sidebar_messages.append({"role": "user", "content": question})
        st.rerun()

# =============================
# MAIN APP INTERFACE (SESSION STATE FIXED)
# =============================
st.markdown('<h1 class="main-title">✈️ TravelAI Premium</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Cultural Tourism | <strong>With Hodophiler(Your true companion for the perfect trip)</strong></p>', unsafe_allow_html=True)

if not df.empty:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Plan Trip", "📜 Itinerary", "🗺️ Explorer", "📊 Insights", 
        "📄 PDF Export", "🎥 Cinematic Video", "📝 Feedback & Analytics"
    ])

    # TAB 1: TRIP PLANNER (SESSION STATE FIXED)
    with tab1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🚀 Plan Your Elite Journey")
        
        # Use NON-CONFLICTING session state keys
        if "trip_data" not in st.session_state:
            st.session_state.trip_data = {}
        
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("### 📍 Journey Details")
            home_country = st.selectbox("🏠 I'm from", options=sorted(df["country"].unique()), 
                                      key="home_country_select")
            travel_countries = [c for c in df["country"].unique() if c != home_country]
            travel_country = st.selectbox("🌍 I want to visit", options=sorted(travel_countries), 
                                        key="travel_country_select")
        
        with col_b:
            st.markdown("### ⚙️ My Profile")
            c1, c2 = st.columns(2)
            budget = c1.selectbox("💰 Budget Level", options=sorted(df["budget_level"].unique()), 
                                key="budget_select")
            duration = c2.selectbox("📅 Trip Length", options=["2 Days (Weekend)", "5 Days", "7 Days"], 
                                  key="duration_select")
            age = st.slider("👤 Age", 18, 90, 30, key="age_slider")
            language = st.selectbox("🌐 Language", options=list(LANGUAGE_MAP.keys()), key="language_select")
        
        interests = st.multiselect("🎯 My Interests", 
            ["Culture","Adventure","Nature","Beaches","Nightlife","Cuisine","Wellness"], 
            default=["Culture","Cuisine"], key="interests_select")
        
        if st.button("🚀 GENERATE ELITE ITINERARY", type="primary", use_container_width=True):
            with st.spinner('🎯 AI planning your perfect adventure...'):
                duration_days = 7 if "7" in duration else 5 if "5" in duration else 2
                best_spot = find_best_destination(home_country, travel_country, budget, 3.0, interests, age, duration_days)
                
                if best_spot is not None:
                    result = generate_consolidated_itinerary(best_spot, age_group(age), duration, language, interests, home_country)
                    
                    # FIXED: Store in NON-CONFLICTING session state
                    st.session_state.trip_data = {
                        "best_spot": best_spot.to_dict(),
                        "result": result,
                        "duration": duration,
                        "home_country": home_country,
                        "language": language,
                        "interests": interests,
                        "budget": budget,
                        "age_group": age_group(age)
                    }
                    
                    st.success(f"✅ Elite itinerary ready for {best_spot['city']}!")
                    st.balloons()
                else:
                    st.error("❌ No destinations match your criteria.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: ITINERARY
    with tab2:
        if "trip_data" in st.session_state and st.session_state.trip_data:
            spot = st.session_state.trip_data["best_spot"]
            st.markdown(f'<h3 style="color: #2563eb;">✨ Elite Itinerary: {spot["city"]}, {spot["country"]}</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⭐ Rating", f"{spot['avg_rating']:.1f}/5")
            col2.metric("💰 Daily Cost", f"${spot['avg_cost_usd_day']:.0f}")
            col3.metric("🌡️ Temperature", f"{spot['avg_temp_monthly']}°C")
            col4.metric("📅 Duration", st.session_state.trip_data["duration"].split()[0])
            
            st.markdown("---")
            st.markdown("### 📋 Your Consolidated Master Plan")
            st.markdown(st.session_state.trip_data["result"], unsafe_allow_html=True)
        else:
            st.info("👆 Plan your dream trip first!")

    # TAB 3: EXPLORER
    with tab3:
        if "trip_data" in st.session_state and st.session_state.trip_data:
            spot = st.session_state.trip_data["best_spot"]
            lat, lng = spot.get('latitude', 0), spot.get('longitude', 0)
            
            st.markdown("### 🗺️ Interactive Destination Map")
            map_url = f"https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3022.1!2d{lng}!3d{lat}!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x0%3A0x0!2z{lat}%2C{lng}!5e0!3m2!1sen!2sin!4v1630000000000"
            st.markdown(f'<div class="map-container"><iframe src="{map_url}" width="100%" height="500" style="border:0;"></iframe></div>', unsafe_allow_html=True)
            
            st.markdown("### 🔗 Instant Research Links")
            col1, col2 = st.columns(2)
            links = [
                (f"🏛️ {spot['city']} Attractions", f"{spot['city']}+best+attractions"),
                (f"🍴 {spot['city']} Restaurants", f"{spot['city']}+best+restaurants"),
                (f"🏨 {spot['city']} Hotels", f"{spot['city']}+luxury+hotels"),
                (f"🚆 {spot['city']} Transport", f"{spot['city']}+public+transport"),
                (f"📸 {spot['city']} Tourist Images", f"{spot['city']}+best+images")
            ]
            for i, (label, query) in enumerate(links):
                url = f"https://google.com/search?q={query}"
                if i < 2:
                    col1.markdown(f"[**{label}**]({url})")
                else:
                    col2.markdown(f"[**{label}**]({url})")
        else:
            st.info("🗺️ Map loads after planning!")

    # TAB 4: INSIGHTS
    with tab4:
        if "trip_data" in st.session_state and st.session_state.trip_data:
            spot = st.session_state.trip_data["best_spot"]
            st.markdown("### 🎯 Perfect Match Analysis")
            
            exp_cols = ["culture","adventure","nature","beaches","nightlife","cuisine","wellness"]
            scores = [spot.get(col, 0) for col in exp_cols]
            
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
            bars = ax.bar(exp_cols, scores, color='#2563eb', alpha=0.8, edgecolor='white')
            ax.set_ylabel("Match Score", fontweight=700)
            ax.set_title(f"Why {spot['city']} is PERFECT for you!", fontweight=800)
            ax.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}', 
                       ha='center', va='bottom', fontweight=600)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("📊 Insights appear after planning!")

    # TAB 5: PDF
    with tab5:
        if "trip_data" in st.session_state and st.session_state.trip_data:
            st.subheader("📄 Professional PDF Export")
            rating = st.slider("⭐ Trip Satisfaction", 1, 5, 5, key="pdf_rating")
            
            pdf_buffer = create_pdf(
                st.session_state.trip_data["result"], 
                pd.Series(st.session_state.trip_data["best_spot"]), 
                rating, 
                st.session_state.trip_data["duration"], 
                st.session_state.trip_data["language"]
            )
            
            st.download_button(
                "📥 DOWNLOAD ELITE GUIDE", 
                pdf_buffer, 
                f"TravelAI_{st.session_state.trip_data['best_spot']['city']}_Guide.pdf", 
                "application/pdf"
            )
        else:
            st.info("✨ Generate itinerary first!")

    # TAB 6: VIDEO
    with tab6:
        if "trip_data" in st.session_state and st.session_state.trip_data:
            st.subheader("🎥 Cinematic Recap Video")
            
            if st.button("🎬 GENERATE VIDEO", type="primary"):
                with st.spinner('🎥 Creating cinematic masterpiece...'):
                    video_buffer = generate_cinematic_gif(
                        pd.Series(st.session_state.trip_data["best_spot"]),
                        st.session_state.trip_data["language"],
                        st.session_state.trip_data["duration"]
                    )
                    st.session_state.video_buffer = video_buffer
                    st.session_state.video_ready = True
                    st.success("🎬 VIDEO READY!")
            
            if hasattr(st.session_state, 'video_ready') and st.session_state.video_ready:
                st.video(st.session_state.video_buffer)
                st.download_button(
                    "💾 DOWNLOAD VIDEO", 
                    st.session_state.video_buffer,
                    f"TravelAI_{st.session_state.trip_data['best_spot']['city']}_Cinematic.gif", 
                    "image/gif"
                )
        else:
            st.info("🎥 Video unlocks after planning!")

    # ===================================================
    # TAB 7: FEEDBACK AGGREGATION & ANALYTICS (NEW)
    # ===================================================
    with tab7:
        st.markdown("## 📝 Feedback Hub & Analytics Dashboard")
        st.markdown("Rate your experience across all features, and explore aggregated insights from all users.")

        # ---- SECTION 1: FEEDBACK FORM ----
        st.markdown("---")
        st.markdown("### 🌟 Submit Your Feedback")

        with st.form("feedback_form", clear_on_submit=True):
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                fb_destination = st.text_input(
                    "📍 Destination Visited",
                    value=st.session_state.trip_data.get("best_spot", {}).get("city", "") if st.session_state.trip_data else "",
                    placeholder="e.g. Paris"
                )
                fb_language = st.selectbox(
                    "🌐 Language Used",
                    options=list(LANGUAGE_MAP.keys()),
                    index=list(LANGUAGE_MAP.keys()).index(
                        st.session_state.trip_data.get("language", "English")
                    ) if st.session_state.trip_data else 0
                )
                fb_season = st.selectbox(
                    "🌤️ Travel Season",
                    ["Spring", "Summer", "Autumn", "Winter"]
                )

            with col_f2:
                fb_itinerary = st.slider("📜 Itinerary Quality", 1, 5, 4, key="fb_itin")
                fb_recommendation = st.slider("🎯 Recommendation Accuracy", 1, 5, 4, key="fb_rec")
                fb_video = st.slider("🎥 Cinematic Video Quality", 1, 5, 4, key="fb_vid")
                fb_chatbot = st.slider("🤖 Hodophiler Chatbot Rating", 1, 5, 4, key="fb_chat")

            fb_comments = st.text_area("💬 Additional Comments (Optional)", placeholder="Tell us more...")

            submitted = st.form_submit_button("✅ SUBMIT FEEDBACK", use_container_width=True)
            if submitted:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "destination": fb_destination,
                    "language": fb_language,
                    "season": fb_season,
                    "itinerary_rating": fb_itinerary,
                    "recommendation_rating": fb_recommendation,
                    "video_rating": fb_video,
                    "chatbot_rating": fb_chatbot,
                    "comments": fb_comments
                }
                save_feedback(entry)
                st.success("🎉 Thank you! Your feedback has been saved.")
                st.balloons()

        # ---- SECTION 2: AGGREGATED INSIGHTS ----
        st.markdown("---")
        st.markdown("### 📊 Aggregated Feedback Analytics")

        df_fb = get_feedback_df()

        if df_fb.empty:
            st.info("📭 No feedback yet. Be the first to submit above!")
        else:
            insights = analyze_feedback_trends(df_fb)

            # Key Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("📋 Total Responses", insights["total_responses"])
            m2.metric("📜 Avg Itinerary", f"{insights['avg_itinerary_rating']}/5")
            m3.metric("🎯 Avg Recommendation", f"{insights['avg_recommendation_rating']}/5")
            m4.metric("🎥 Avg Video", f"{insights['avg_video_rating']}/5")
            m5.metric("🤖 Avg Chatbot", f"{insights['avg_chatbot_rating']}/5")

            st.markdown("---")

            col_v1, col_v2 = st.columns(2)

            # Chart 1: Average Ratings by Feature (Bar)
            with col_v1:
                st.markdown("#### ⭐ Average Ratings by Feature")
                rating_data = {
                    "Feature": ["Itinerary", "Recommendation", "Video", "Chatbot"],
                    "Avg Rating": [
                        insights["avg_itinerary_rating"],
                        insights["avg_recommendation_rating"],
                        insights["avg_video_rating"],
                        insights["avg_chatbot_rating"]
                    ]
                }
                fig_bar = px.bar(
                    rating_data, x="Feature", y="Avg Rating",
                    color="Avg Rating", color_continuous_scale="blues",
                    range_y=[0, 5], text="Avg Rating",
                    title="Feature-wise Average Ratings"
                )
                fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_bar.update_layout(coloraxis_showscale=False, plot_bgcolor='white')
                st.plotly_chart(fig_bar, use_container_width=True)

            # Chart 2: Top Destinations by Feedback Count (Pie)
            with col_v2:
                st.markdown("#### 🏙️ Top Rated Destinations")
                if "destination" in df_fb.columns and df_fb["destination"].notna().any():
                    dest_counts = df_fb["destination"].value_counts().reset_index()
                    dest_counts.columns = ["Destination", "Count"]
                    fig_pie = px.pie(
                        dest_counts, values="Count", names="Destination",
                        title="Feedback by Destination",
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No destination data yet.")

            col_v3, col_v4 = st.columns(2)

            # Chart 3: Language Preferences (Bar)
            with col_v3:
                st.markdown("#### 🌐 Language Preferences")
                if "language" in df_fb.columns:
                    lang_counts = df_fb["language"].value_counts().reset_index()
                    lang_counts.columns = ["Language", "Count"]
                    fig_lang = px.bar(
                        lang_counts, x="Language", y="Count",
                        color="Count", color_continuous_scale="purples",
                        title="Most Used Languages"
                    )
                    fig_lang.update_layout(coloraxis_showscale=False, plot_bgcolor='white')
                    st.plotly_chart(fig_lang, use_container_width=True)

            # Chart 4: Season Popularity (Donut)
            with col_v4:
                st.markdown("#### 🌤️ Travel Season Popularity")
                if "season" in df_fb.columns:
                    season_counts = df_fb["season"].value_counts().reset_index()
                    season_counts.columns = ["Season", "Count"]
                    fig_season = px.pie(
                        season_counts, values="Count", names="Season",
                        hole=0.45, title="Preferred Travel Seasons",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_season, use_container_width=True)

            # Chart 5: Ratings Over Time (Line)
            st.markdown("#### 📈 Feedback Trends Over Time")
            df_fb_sorted = df_fb.sort_values("timestamp")
            df_fb_sorted["date"] = df_fb_sorted["timestamp"].dt.date
            daily_avg = df_fb_sorted.groupby("date")[
                ["itinerary_rating", "recommendation_rating", "video_rating", "chatbot_rating"]
            ].mean().reset_index()

            fig_line = px.line(
                daily_avg, x="date",
                y=["itinerary_rating", "recommendation_rating", "video_rating", "chatbot_rating"],
                labels={"value": "Avg Rating", "variable": "Feature", "date": "Date"},
                title="Daily Average Ratings Trend",
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_line.update_layout(plot_bgcolor='white', yaxis_range=[0, 5])
            st.plotly_chart(fig_line, use_container_width=True)

            # Chart 6: Radar Chart - Overall Feature Balance
            st.markdown("#### 🕸️ Feature Balance Radar")
            radar_vals = [
                insights["avg_itinerary_rating"],
                insights["avg_recommendation_rating"],
                insights["avg_video_rating"],
                insights["avg_chatbot_rating"],
                insights["avg_itinerary_rating"]   # close the loop
            ]
            radar_cats = ["Itinerary", "Recommendation", "Video", "Chatbot", "Itinerary"]
            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals,
                theta=radar_cats,
                fill='toself',
                line_color='#2563eb',
                fillcolor='rgba(37,99,235,0.2)'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                title="Overall Platform Performance Radar",
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # ---- SECTION 3: ADAPTIVE IMPROVEMENT NOTICE ----
            st.markdown("---")
            st.markdown("### 🔄 Adaptive Improvement Engine")
            st.markdown("""
            The recommendation engine **automatically adapts** based on your collective feedback:
            - Destinations with average itinerary ratings **≥ 4.0** receive a **+15 score boost** in future recommendations.
            - Destinations with average ratings **≥ 3.0** receive a **+5 score boost**.
            - This ensures popular, well-loved destinations surface more often for future travelers.
            """)

            # Show top boosted destinations
            if "destination" in df_fb.columns:
                dest_avg = df_fb.groupby("destination")["itinerary_rating"].mean().reset_index()
                dest_avg.columns = ["Destination", "Avg Itinerary Rating"]
                dest_avg["Adaptive Boost"] = dest_avg["Avg Itinerary Rating"].apply(
                    lambda x: "+15 🚀" if x >= 4.0 else ("+5 📈" if x >= 3.0 else "0")
                )
                dest_avg = dest_avg.sort_values("Avg Itinerary Rating", ascending=False)
                st.dataframe(dest_avg, use_container_width=True)

            # Raw feedback table (expandable)
            with st.expander("🗂️ View Raw Feedback Data"):
                st.dataframe(df_fb.drop(columns=["comments"], errors="ignore"), use_container_width=True)

else:
    st.error("❌ Place 'master_dataset_week2.csv' in the same folder!")

# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <h3>🎓 CAPSTONE PROJECT </h3>
    <p><strong> | ❤️A Journey to the Perfect Trip by Aryan & Dibyajyoti❤️ </strong></p>
</div>
""", unsafe_allow_html=True)
