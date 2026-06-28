import streamlit as st # Enhanced version
import pickle
import pandas as pd
import random
from src.remove_ import remove
from src.recommender import RecommenderEngine

# Page Config
st.set_page_config(page_title="SmartPix | Mobile Recommender", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# Load Data
@st.cache_resource(show_spinner="Loading SmartPix Engine...")
def load_data():
    df = pickle.load(file=open(file=r'src/model/dataframe.pkl', mode='rb'))
    similarity = pickle.load(file=open(file=r'src/model/similarity.pkl', mode='rb'))
    
    # Merge the new structured 26-column data safely
    try:
        # Use a proper merge on the 'name' column to prevent scrambled data
        structured_df = pd.read_csv('data/processed/smartphones_structured_26cols.csv')
        # Deduplicate before merging to avoid cartesian-product row explosion
        name_col = 'name' if 'name' in structured_df.columns else 'model_name'
        structured_df = structured_df.drop_duplicates(subset=[name_col], keep='first')
        
        # Identify columns in structured_df that are not in df (except 'name')
        new_cols = [c for c in structured_df.columns if c not in df.columns or c == 'name']
        
        # If both have 'name', we can merge. If not, we fall back to index-based but reset both.
        if 'name' in df.columns and ('model_name' in structured_df.columns or 'name' in structured_df.columns):
            s_name_col = 'name' if 'name' in structured_df.columns else 'model_name'
            df = df.merge(structured_df[new_cols].rename(columns={s_name_col: 'name'}), on='name', how='left')
        else:
            df = df.reset_index(drop=True)
            structured_df = structured_df.reset_index(drop=True)
            for col in structured_df.columns:
                if col not in df.columns:
                    df[col] = structured_df[col]
    except Exception as e:
        st.error(f"Data Sync Error: {e}")
        
    engine = RecommenderEngine(df, similarity)
    return df, similarity, engine

df, similarity, engine = load_data()

# Custom CSS for Premium UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Outfit:wght@300;500;700&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
        --space-bg: #030712;
    }

    /* Force transparent backgrounds for Streamlit containers to reveal the starfield */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
        background: transparent !important;
    }

    [data-testid="stMainViewContainer"] {
        background: transparent !important;
    }

    .stApp {
        background: radial-gradient(circle at top right, #0f172a 0%, #020617 100%) !important;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
        overflow-x: hidden;
    }

    /* --- FLOATING APP ICONS --- */
    .floating-icons-container {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: 0;
        pointer-events: none;
        overflow: hidden;
    }

    .floating-icon {
        position: absolute;
        bottom: -150px;
        animation: floatUp linear infinite;
        opacity: 0.4;
        transition: opacity 0.5s;
        will-change: transform;
    }

    .app-icon-bg {
        width: 64px;
        height: 64px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.1);
    }

    @keyframes floatUp {
        0% { transform: translateY(0) rotate(-15deg); opacity: 0; }
        5% { opacity: 0.4; }
        95% { opacity: 0.4; }
        100% { transform: translateY(-120vh) rotate(15deg); opacity: 0; }
    }

    /* Nebula effect as a static overlay if needed, but keeping it simple for now */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.05) 0%, transparent 80%) !important;
    }

    h1, h2, h3, .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(to right, #fff, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(129, 140, 248, 0.3);
    }

    /* Glassmorphism Card */
    .mobile-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 24px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .mobile-card:hover {
        transform: translateY(-12px) scale(1.03);
        border-color: #818cf8;
        box-shadow: 0 0 40px rgba(99, 102, 241, 0.2);
    }

    /* Landing Page Styles */
    .hero-section {
        padding: 40px 20px;
        text-align: center;
        position: relative;
        z-index: 1;
    }

    .hero-title {
        font-size: 6rem;
        margin-bottom: 1.5rem;
        letter-spacing: -2px;
        animation: titleGlow 4s ease-in-out infinite alternate;
    }

    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 10px rgba(255,255,255,0.2)); }
        to { filter: drop-shadow(0 0 30px rgba(129, 140, 248, 0.6)); }
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        max-width: 800px;
        margin: 0 auto 2.5rem;
    }

    /* Global Text Contrast Fixes */
    .stApp p, .stApp span, .stApp label, .stApp .stSlider * {
        color: #f8fafc !important;
    }

    /* Enhanced selectbox styling with comprehensive selectors */
    .stSelectbox > div > div > div {
        background-color: rgba(129, 140, 248, 0.1) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
    }
    
    .stSelectbox > div > div > div:hover {
        background-color: rgba(129, 140, 248, 0.2) !important;
        border: 1px solid rgba(129, 140, 248, 0.5) !important;
    }
    
    /* Target the dropdown menu items */
    .stSelectbox [data-baseweb="menu"] li:hover {
        background-color: rgba(129, 140, 248, 0.3) !important;
        color: #f8fafc !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #818cf8 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Alternative selectors for different Streamlit versions */
    div[data-testid="stSelectboxSelectControl"] {
        background-color: rgba(129, 140, 248, 0.1) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
    }
    
    div[data-testid="stSelectboxSelectControl"]:hover {
        background-color: rgba(129, 140, 248, 0.2) !important;
        border: 1px solid rgba(129, 140, 248, 0.5) !important;
    }
    
    /* BaseWeb select styling */
    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }
    
    div[data-baseweb="select"] div[role="listbox"] div[role="option"][aria-selected="true"] {
        background-color: #818cf8 !important;
        color: white !important;
    }
    
    div[data-baseweb="select"] div[role="listbox"] div[role="option"]:hover {
        background-color: rgba(129, 140, 248, 0.3) !important;
        color: #f8fafc !important;
    }
    
    .stSelectbox label, .stSlider label {
        color: #a5b4fc !important;
        font-weight: 600 !important;
    }

    .stApp small {
        color: #94a3b8 !important;
    }

    /* Custom Button */
    .stButton > button {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Hide Deploy button, three-dot menu, and footer */
    [data-testid="stDeployButton"],
    .stAppDeployButton,
    [data-testid="stToolbar"],
    [data-testid="stToolbarActions"],
    footer,
    [data-testid="stFooter"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide heading anchor links added by Streamlit */
    h1 a, h2 a, h3 a,
    [data-testid="stHeadingWithActionElements"] a,
    .stHeadingWithActionElements a {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# remove()

# Generate 20 random floating app icons
def get_floating_icons():
    import random
    random.seed(42)  # Fixed seed to prevent jumpy refresh on every streamlit rerun
    icons = [
        {"bg": "linear-gradient(45deg, #f09433, #dc2743, #bc1888)", "icon": "📷"}, # Insta
        {"bg": "#25D366", "icon": "💬"}, # WA
        {"bg": "#1877F2", "icon": "👥"}, # FB
        {"bg": "#FF0000", "icon": "▶️"}, # YT
        {"bg": "#1DB954", "icon": "🎵"}, # Spotify
        {"bg": "#1DA1F2", "icon": "🐦"}, # Twitter
        {"bg": "#FFFFFF", "icon": "📧"}, # Mail
        {"bg": "#FFFC00", "icon": "👻"}, # Snap
        {"bg": "#000000", "icon": "🎬"}, # Netflix
        {"bg": "#FF9900", "icon": "🛒"}, # Amazon
        {"bg": "#0088cc", "icon": "✈️"}, # Telegram
        {"bg": "#FF4500", "icon": "🤖"}, # Reddit
        {"bg": "#5865F2", "icon": "🎮"}, # Discord
        {"bg": "#E60023", "icon": "📌"}, # Pinterest
        {"bg": "#0077B5", "icon": "💼"}, # LinkedIn
        {"bg": "#000000", "icon": "📱"}, # Apple
        {"bg": "#34A853", "icon": "🗺️"}, # Maps
        {"bg": "#EA4335", "icon": "🔍"}, # Google
        {"bg": "#4285F4", "icon": "☁️"}, # Cloud
        {"bg": "#FDB813", "icon": "☀️"}  # Weather
    ]
    
    html = '<div class="floating-icons-container">'
    for i in range(20):
        icon = icons[i] # exactly 20 distinct icons
        left = (i * 4.8) + random.uniform(0, 4) # Distribute evenly across width
        duration = random.randint(40, 90) # Slow, but visibly moving (40-90 seconds)
        delay = random.randint(-350, 0)
        scale = random.uniform(0.6, 1.2)
        
        html += f'''
<div class="floating-icon" style="left: {left}%; animation-duration: {duration}s; animation-delay: {delay}s;">
<div class="app-icon-bg" style="background: {icon['bg']}; transform: scale({scale});">
<span style="font-size: 2rem;">{icon['icon']}</span>
</div>
</div>
'''
    html += '</div>'
    return html

st.markdown(get_floating_icons(), unsafe_allow_html=True)

def display_mobiles(recommended_df, title="Recommended Mobiles", persona_idx=None, active_specs=None):
    import math

    if recommended_df.empty:
        st.warning("No mobiles found matching your criteria.")
        return

    st.markdown(f"<h2 style='margin-bottom: 2rem;'>{title}</h2>", unsafe_allow_html=True)

    def safe_num(val, min_v=0, max_v=99999):
        try:
            v = float(val)
            if math.isnan(v) or math.isinf(v): return 0
            return v if min_v <= v <= max_v else 0
        except:
            return 0

    def fmt(v):
        return str(int(v)) if v == int(v) else str(round(v, 1))

    def get_str(row, col):
        val = row.get(col, '')
        return str(val).strip() if val and str(val).strip() not in ('nan', 'None', '') else ''

    # Persona → which spec tags to display
    PERSONA_SPECS = {
        1:  ['processor', 'ram',     'battery', 'refresh_rate', 'fast_charge', 'screen'],      # Gaming
        2:  ['ram',       'battery', 'storage',  'processor',   'screen',      'price'],       # Daily Use
        3:  ['camera',    'f_camera','ois',      'battery',     'storage',     'ram', 'display'], # Photography
        4:  ['ram',       'battery', 'storage',  'processor',   'screen',      'price'],       # Business
        5:  ['display',   'stereo',  'screen',   'battery',     'fast_charge', 'ram'],         # Entertainment
        6:  ['battery',   'fast_charge', 'processor', 'storage','screen',      'ram'],         # Battery
        7:  ['f_camera',  'camera',  'storage',  'ram',         'battery',     'fast_charge'], # Social Media
        8:  ['price',     'ram',     'battery',  'storage',     'camera',      'screen'],      # Students
        9:  ['battery',   'storage', 'screen',   'price',       'ram',         'camera'],      # Seniors
        10: ['battery',   'build',   'screen',   '5g',          'processor',   'storage'],     # Travel
        11: ['processor', 'ram',     'refresh_rate', 'fast_charge', 'screen',  'camera', 'battery'], # Tech Enthusiasts
        12: ['build',     'screen',  'storage',  'ram',         'display',     'battery'],     # Design
    }
    if active_specs is not None:
        spec_keys = active_specs
    else:
        spec_keys = PERSONA_SPECS.get(persona_idx, ['ram', 'storage', 'battery', 'camera'])

    def build_specs(row):
        parts = []
        for key in spec_keys:
            if key == 'ram':
                v = safe_num(row.get('ram_gb'), 1, 24) or safe_num(row.get('ram'), 1, 24)
                if v: parts.append(f"🧠 RAM: {fmt(v)}GB")
            elif key == 'storage':
                v = safe_num(row.get('storage_gb'), 1, 1024) or safe_num(row.get('storage'), 1, 1024)
                if v: parts.append(f"💾 Storage: {fmt(v)}GB")
            elif key == 'battery':
                v = safe_num(row.get('battery_mah'), 1000, 10000) or safe_num(row.get('battery'), 1000, 10000)
                if v: parts.append(f"🔋 {fmt(v)}mAh")
            elif key == 'camera':
                v = safe_num(row.get('rear_camera_mp'), 1, 200) or safe_num(row.get('camera'), 1, 200)
                if v: parts.append(f"📷 {fmt(v)}MP")
            elif key == 'f_camera':
                v = safe_num(row.get('front_camera_mp'), 1, 100)
                if v: parts.append(f"🤳 {fmt(v)}MP")
            elif key == 'ois':
                v = get_str(row, 'ois')
                if v and v.lower() == 'yes': parts.append("✅ OIS")
            elif key == 'processor':
                v = get_str(row, 'processor')
                if v: parts.append(f"⚡ {v[:22]}")
            elif key == 'refresh_rate':
                v = safe_num(row.get('refresh_rate'), 60, 240)
                if v: parts.append(f"🖥️ {fmt(v)}Hz")
            elif key == 'display':
                v = get_str(row, 'display_type')
                if v: parts.append(f"🖥️ {v}")
                elif 'amoled' in str(row.get('corpus','')).lower(): parts.append("🖥️ AMOLED")
            elif key == 'stereo':
                v = get_str(row, 'stereo_speakers')
                if v and v.lower() == 'yes': parts.append("🔊 Stereo")
                elif 'stereo' in str(row.get('corpus','')).lower(): parts.append("🔊 Stereo")
            elif key == 'screen':
                v = safe_num(row.get('screen_size'), 4, 8) or safe_num(row.get('screen'), 4, 8)
                if v: parts.append(f"📐 {fmt(v)}\"")
            elif key == 'fast_charge':
                v = safe_num(row.get('fast_charging_watt'), 5, 240)
                if v: parts.append(f"⚡ {fmt(v)}W")
                elif 'fast charg' in str(row.get('corpus','')).lower(): parts.append("⚡ Fast Charge")
            elif key == 'build':
                if 'gorilla' in str(row.get('corpus','')).lower(): parts.append("🛡️ Gorilla Glass")
                elif 'corning' in str(row.get('corpus','')).lower(): parts.append("🛡️ Corning Glass")
            elif key == '5g':
                v = get_str(row, '5g')
                if v in ('Yes', 'True', '1', 'yes'): parts.append("📶 5G")
                elif '5g' in str(row.get('corpus','')).lower(): parts.append("📶 5G")
            elif key == 'price':
                v = safe_num(row.get('price_numeric'), 1000, 500000)
                if v: parts.append(f"💰 ₹{fmt(v)}")
        return "  |  ".join(parts) if parts else "Specs N/A"

    # Split into chunks of 4
    for i in range(0, len(recommended_df), 4):
        cols = st.columns(4)
        chunk = recommended_df.iloc[i:i+4]
        for idx, (index, row) in enumerate(chunk.iterrows()):
            with cols[idx]:
                specs = build_specs(row)
                p_num = safe_num(row.get('price_numeric'), 1)
                if p_num > 0:
                    price_display = f"₹{int(p_num):,}"
                else:
                    raw_price = str(row.get('price', 'Price N/A'))
                    price_display = raw_price if '₹' in raw_price else f"₹{raw_price}"
                
                try:
                    rating_val = float(row.get('ratings', 0))
                    rating_display = f"⭐ {rating_val:.1f}"
                except:
                    rating_display = f"⭐ {row.get('ratings', 0)}"

                import urllib.parse
                web_img = f"https://tse2.mm.bing.net/th?q={urllib.parse.quote_plus(row['name'] + ' smartphone')}"

                st.markdown(f"""
                    <div class="mobile-card">
                        <img src="{web_img}" class="mobile-img" style="width: 100%; height: 200px; object-fit: contain;">
                        <div style="font-weight: 700; margin-bottom: 6px; color: #f8fafc; font-size:0.95rem;">{row['name']}</div>
                        <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 10px; line-height:1.6;">{specs}</div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #818cf8;">{price_display}</div>
                        <div style="font-size: 0.8rem; margin-top: 5px;">{rating_display}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

def render_landing_page():
    st.markdown("""
        <div class="hero-section" style="position: relative; z-index: 10;">
            <h1 class="hero-title" style="font-size: 4rem; margin-bottom: 0.5rem;">SmartPix</h1>
            <h3 style="font-size: 2rem; font-weight: 500; margin-bottom: 1.5rem; color: #a5b4fc;">The Smart Way to Choose Smart Phone</h3>
            <div style="max-width: 800px; margin: 0 auto;">
                <p class="hero-subtitle" style="line-height: 1.6; font-size: 1.2rem;">
                    Experience next-gen recommendations tailored to your digital lifestyle. SmartPix analyzes your preferences, usage patterns, and budget to deliver the perfect smartphone—quickly, intelligently, and without the confusion of endless comparisons.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Launch Discovery", use_container_width=True):
            st.session_state.page = 'app'
            st.rerun()

def render_recommender_page():
    # Sidebar for Recommendation Mode
    st.sidebar.title("Recommendation Settings")
    mode = st.sidebar.selectbox(
        "Select Recommendation Mode",
        ["Rule-Based", "Content-Based", "Weighted Scoring", "KNN"]
    )

    st.markdown("<h1 class='main-title'>SmartPix Recommendations</h1>", unsafe_allow_html=True)

    if mode == "Rule-Based":
        st.sidebar.subheader("Filters")
        filter_type = st.sidebar.radio("Filter Type", ["Use Case / Persona", "Custom Filters"])
        
        if filter_type == "Use Case / Persona":
            personas = {
                1: "🎮 1. Gaming",
                2: "📱 2. Normal / Daily Use",
                3: "📸 3. Photography",
                4: "💼 4. Business / Productivity",
                5: "🎬 5. Entertainment / Media Consumption",
                6: "🔋 6. Battery-Focused Users",
                7: "📶 7. Social Media / Content Creators",
                8: "🎓 8. Students",
                9: "👴 9. Senior Citizens / Simple Use",
                10: "📡 10. Travel / Outdoor Users",
                11: "🧑💻 11. Tech Enthusiasts",
                12: "🎨 12. Design / Aesthetic Lovers"
            }
            persona_options = list(personas.values())
            selected_persona_name = st.sidebar.selectbox("Select Your Persona", persona_options)
            selected_idx = list(personas.keys())[persona_options.index(selected_persona_name)]
            
            st.sidebar.subheader("Sort By")
            persona_sort_option = st.sidebar.selectbox(
                "Sort Results By",
                ["Relevance", "Price: Low to High", "Price: High to Low", 
                 "Rating: High to Low", "Popularity (Rating Count)"]
            )
            
            if st.button('Find Best Mobiles'):
                results = engine.persona_based(selected_idx, sort_option=persona_sort_option)
                display_mobiles(results, f"Top Picks for: {selected_persona_name}", persona_idx=selected_idx)
                
        else:
            st.sidebar.subheader("📊 Basic Specs")
            # Enhanced Price Range Filter
            min_budget, max_budget = st.sidebar.slider(
                "💰 Price Range (₹)", 
                min_value=5000, 
                max_value=200000, 
                value=(10000, 50000), 
                step=5000
            )
            
            ram = st.sidebar.selectbox("🧠 Min RAM (GB)", [0, 2, 4, 6, 8, 12, 16])
            storage = st.sidebar.selectbox("💾 Min Storage (GB)", [0, 32, 64, 128, 256, 512])

            st.sidebar.subheader("📷 Camera")
            camera = st.sidebar.selectbox("📷 Min Rear Camera (MP)", [0, 12, 32, 48, 64, 108, 200])
            front_camera = st.sidebar.selectbox("🤳 Min Front Camera (MP)", [0, 8, 16, 32, 50])

            st.sidebar.subheader("⚡ Performance & Battery")
            battery = st.sidebar.selectbox("🔋 Min Battery (mAh)", [0, 3000, 4000, 4500, 5000, 6000, 7000])
            fast_charging = st.sidebar.checkbox("⚡ Fast Charging Required")
            refresh_rate = st.sidebar.selectbox("🖥️ Min Refresh Rate (Hz)", [0, 60, 90, 120, 144, 165])

            st.sidebar.subheader("📡 Connectivity")
            network = st.sidebar.radio("📶 Network", ["Any", "5G Only", "4G Only"])

            st.sidebar.subheader("📱 Brand")
            # Multi-select Brand Filter
            available_brands = ["Samsung", "Apple", "OnePlus", "Xiaomi", "Redmi",
                              "POCO", "Realme", "OPPO", "Vivo", "iQOO", "Motorola",
                              "Infinix", "Tecno", "Nokia", "Google", "Nothing"]
            selected_brands = st.sidebar.multiselect(
                "Select Brands (leave empty for Any)", 
                available_brands,
                default=[]
            )
            
            st.sidebar.subheader("⭐ Rating")
            # Single-select Rating Filter
            min_rating = st.sidebar.selectbox(
                "Minimum Rating",
                ["Any", "3.0+ Stars", "3.5+ Stars", "4.0+ Stars", "4.5+ Stars"],
                index=0
            )
            
            st.sidebar.subheader("🖥️ Display Types")
            # Multi-select Display Type Filter
            display_type_options = ["AMOLED", "Super AMOLED", "OLED", "LCD", "IPS"]
            selected_display_types = st.sidebar.multiselect(
                "Select Display Types (leave empty for Any)",
                display_type_options,
                default=[]
            )
            
            st.sidebar.subheader("📋 Sort By")
            # Sorting Options
            sort_option = st.sidebar.selectbox(
                "Sort Results By",
                ["Relevance", "Price: Low to High", "Price: High to Low", 
                 "Rating: High to Low", "Popularity (Rating Count)"]
            )

            # Action Buttons
            col1, col2 = st.sidebar.columns(2)
            with col1:
                find_button = st.button('🔍 Find Best Mobiles', use_container_width=True)
            with col2:
                reset_button = st.button('🔄 Reset Filters', use_container_width=True)
            
            # Reset functionality
            if reset_button:
                st.session_state.clear()
                st.rerun()
            
            if find_button:
                # Convert single rating selection to numeric value
                rating_value = 0.0
                if "4.5" in min_rating: rating_value = 4.5
                elif "4.0" in min_rating: rating_value = 4.0
                elif "3.5" in min_rating: rating_value = 3.5
                elif "3.0" in min_rating: rating_value = 3.0
                
                # Display selected filters
                st.markdown("### Active Filters")
                filter_cols = st.columns(4)
                
                with filter_cols[0]:
                    st.metric("Price Range", f"Rs.{min_budget:,} - Rs.{max_budget:,}")
                
                with filter_cols[1]:
                    if selected_brands:
                        st.metric("Brands", f"{len(selected_brands)} selected")
                    else:
                        st.metric("Brands", "Any")
                
                with filter_cols[2]:
                    st.metric("Min Rating", min_rating)
                
                with filter_cols[3]:
                    st.metric("Sort By", sort_option)
                
                # Additional filter details
                active_filters = []
                if ram > 0: active_filters.append(f"RAM: {ram}GB+")
                if storage > 0: active_filters.append(f"Storage: {storage}GB+")
                if battery > 0: active_filters.append(f"Battery: {battery}mAh+")
                if camera > 0: active_filters.append(f"Camera: {camera}MP+")
                if front_camera > 0: active_filters.append(f"Front Camera: {front_camera}MP+")
                if fast_charging: active_filters.append("Fast Charging")
                if refresh_rate > 0: active_filters.append(f"Refresh Rate: {refresh_rate}Hz+")
                if network != "Any": active_filters.append(f"Network: {network}")
                if selected_display_types: active_filters.append(f"Display: {', '.join(selected_display_types[:2])}{'...' if len(selected_display_types) > 2 else ''}")
                
                if active_filters:
                    st.markdown("**Additional Filters:** " + " | ".join(active_filters))
                
                st.markdown("---")
                
                # Get results with enhanced filtering
                results = engine.rule_based_enhanced(
                    min_budget=min_budget, max_budget=max_budget,
                    min_ram=ram, min_storage=storage,
                    min_battery=battery, min_camera=camera,
                    min_front_camera=front_camera, brands=selected_brands,
                    network=network, display_types=selected_display_types,
                    fast_charging=fast_charging, min_refresh_rate=refresh_rate,
                    min_rating=rating_value,
                    sort_option=sort_option
                )
                
                # Build title with selected filters
                title_parts = [f"Rs.{min_budget:,} - Rs.{max_budget:,}"]
                if selected_brands: title_parts.append(", ".join(selected_brands[:3]) + ("..." if len(selected_brands) > 3 else ""))
                if network != "Any": title_parts.append(network)
                if selected_display_types: title_parts.append(", ".join(selected_display_types[:2]) + ("..." if len(selected_display_types) > 2 else ""))
                if min_rating != "Any": title_parts.append(min_rating)

                # Build active specs based on what the user actually filtered
                active_specs = []
                if ram > 0: active_specs.append('ram')
                if storage > 0: active_specs.append('storage')
                if camera > 0: active_specs.append('camera')
                if front_camera > 0: active_specs.append('f_camera')
                if battery > 0: active_specs.append('battery')
                if fast_charging: active_specs.append('fast_charge')
                if refresh_rate > 0: active_specs.append('refresh_rate')
                if selected_display_types:
                    active_specs.append('display')
                    active_specs.append('screen')
                if network == "5G Only": active_specs.append('5g')
                if not active_specs:
                    active_specs = ['ram', 'storage', 'battery', 'camera']

                # Display results with enhanced message
                result_title = "Recommended Phones Based on Your Preferences"
                if not results.empty:
                    st.info(f"✨ Found {len(results)} phones matching your criteria")
                    display_mobiles(results, f"{result_title} — " + " | ".join(title_parts), active_specs=active_specs)
                else:
                    st.warning("No phones found matching your criteria. Try adjusting filters.")


    elif mode == "Content-Based":
        st.sidebar.subheader("Budget")
        # Enhanced Price Range Filter
        cb_min_budget, cb_max_budget = st.sidebar.slider(
            "Price Range (Rs.)", 
            min_value=5000, 
            max_value=200000, 
            value=(10000, 50000), 
            step=5000
        )

        st.sidebar.subheader("Usage Type")
        usage_map = {
            "Gaming": "gaming",
            "Photography": "photography",
            "Battery Life": "battery",
            "Entertainment": "entertainment",
            "Normal Use": "normal",
        }
        usage_label = st.sidebar.selectbox("Primary Usage", list(usage_map.keys()))
        usage_type = usage_map[usage_label]

        st.sidebar.subheader("Feature Preferences")
        perf_pref    = st.sidebar.select_slider("Performance", ["low", "medium", "high"], value="medium")
        camera_pref  = st.sidebar.select_slider("Camera",      ["low", "medium", "high"], value="medium")
        battery_pref = st.sidebar.select_slider("Battery",     ["low", "medium", "high"], value="medium")
        display_pref = st.sidebar.select_slider("Display",     ["low", "medium", "high"], value="medium")
        
        st.sidebar.subheader("Sort By")
        cb_sort_option = st.sidebar.selectbox(
            "Sort Results By",
            ["Relevance", "Price: Low to High", "Price: High to Low", 
             "Rating: High to Low", "Popularity (Rating Count)"]
        )

        if st.button("Find My Perfect Phone"):
            results, reasons = engine.preference_based(
                min_budget=cb_min_budget, max_budget=cb_max_budget,
                usage_type=usage_type, perf_pref=perf_pref, 
                camera_pref=camera_pref, battery_pref=battery_pref, 
                display_pref=display_pref, sort_option=cb_sort_option
            )
            st.session_state.cb_results = (results, reasons)

        if 'cb_results' in st.session_state:
            results, reasons = st.session_state.cb_results
            if results.empty:
                st.warning(f"No phones found in your price range. Try adjusting your budget.")
            else:
                import math
                st.info(f" Found {len(results)} matches for your profile")
                st.markdown(f"<h2 style='margin-bottom:1rem;'> Your Perfect Matches</h2>", unsafe_allow_html=True)

                def safe_score(v):
                    try:
                        f = float(v)
                        return f if not math.isnan(f) else 0
                    except: return 0

                for i in range(0, len(results), 4):
                    cols = st.columns(4)
                    chunk = results.iloc[i:i+4]
                    reason_chunk = reasons.iloc[i:i+4]
                    for idx, ((_, row), reason) in enumerate(zip(chunk.iterrows(), reason_chunk)):
                        with cols[idx]:
                            p_score  = safe_score(row.get('performance_score', 0))
                            c_score  = safe_score(row.get('camera_score', 0))
                            b_score  = safe_score(row.get('battery_score', 0))
                            d_score  = safe_score(row.get('display_score', 0))
                            price_v  = safe_score(row.get('price_numeric', 0))
                            price_s  = f"₹{int(price_v):,}" if price_v else row.get('price','N/A')

                            def bar(score, color="#818cf8"):
                                pct = int(score * 10)
                                return f"<div style='background:rgba(255,255,255,0.08);border-radius:4px;height:6px;margin:2px 0'><div style='width:{pct}%;background:{color};height:6px;border-radius:4px'></div></div>"

                            try:
                                r_val = float(row.get('ratings', 0))
                                r_disp = f"⭐ {r_val:.1f}"
                            except:
                                r_disp = f"⭐ {row.get('ratings', 0)}"

                            import urllib.parse
                            web_img = f"https://tse2.mm.bing.net/th?q={urllib.parse.quote_plus(row['name'] + ' smartphone')}"

                            st.markdown(f"""
                                <div class="mobile-card">
                                    <img src="{web_img}" style="width:100%;height:180px;object-fit:contain;">
                                    <div style="font-weight:700;font-size:0.9rem;color:#f8fafc;margin-bottom:6px;">{row['name']}</div>
                                    <div style="font-size:0.75rem;color:#94a3b8;">
                                        ⚡ Perf {p_score}/10 {bar(p_score,'#818cf8')}
                                        📷 Cam {c_score}/10  {bar(c_score,'#34d399')}
                                        🔋 Bat {b_score}/10  {bar(b_score,'#facc15')}
                                        🖥️ Disp {d_score}/10 {bar(d_score,'#f472b6')}
                                    </div>
                                    <div style="font-size:1rem;font-weight:800;color:#818cf8;margin-top:8px;">{price_s}</div>
                                    <div style="font-size:0.8rem;margin-top:4px;">{r_disp}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)


    
    elif mode == "Weighted Scoring":
        st.sidebar.subheader("Budget & Feature Weights")
        st.sidebar.caption("Set a slider to **1.0** for highest priority, or **0.0** to ignore the feature.")
        w_budget = st.sidebar.slider("Max Budget (₹)", 5000, 200000, 200000, step=5000)
        w_perf = st.sidebar.slider("Processor Performance", 0.0, 1.0, 0.5)
        w_ram = st.sidebar.slider("RAM", 0.0, 1.0, 0.5)
        w_storage = st.sidebar.slider("Storage", 0.0, 1.0, 0.5)
        w_rating = st.sidebar.slider("Rating", 0.0, 1.0, 0.5)
        w_battery = st.sidebar.slider("Battery Capacity", 0.0, 1.0, 0.5)
        w_camera = st.sidebar.slider("Camera Quality", 0.0, 1.0, 0.5)
        
        weights = {
            'performance': w_perf, 'ram': w_ram, 'storage': w_storage, 
            'rating': w_rating, 'battery': w_battery, 'camera': w_camera
        }
        
        st.sidebar.subheader("Sort By")
        ws_sort_option = st.sidebar.selectbox(
            "Sort Results By",
            ["Relevance", "Price: Low to High", "Price: High to Low", 
             "Rating: High to Low", "Popularity (Rating Count)"]
        )
        
        if st.button('Calculate Best Value'):
            results = engine.weighted_scoring(weights, max_budget=w_budget, sort_option=ws_sort_option)
            display_mobiles(results, f"Best Value under Rs.{w_budget:,}")

    elif mode == "KNN":
        mobiles = df['name'].values
        selected_mobile = st.selectbox(label='Select a reference mobile', options=mobiles)
        if st.button('Find Nearest Neighbors'):
            results = engine.knn_recommend(selected_mobile)
            display_mobiles(results, f"Nearest Neighbors to {selected_mobile}")

    # Footer removed

# Main Navigation
if st.session_state.page == 'landing':
    render_landing_page()
else:
    render_recommender_page()
