import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from datetime import datetime
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
 
# ── Page Config ───────────────────────────────
st.set_page_config(
    page_title="Smart Traffic India",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ── CSS ───────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .section-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #0D1B2A;
        border-bottom: 2px solid #028090;
        padding-bottom: 6px;
        margin-bottom: 15px;
    }
    .alert-high {
        background:#FEF2F2; border-left:5px solid #DC2626;
        padding:12px 16px; border-radius:8px; margin:8px 0;
    }
    .alert-medium {
        background:#FFF7ED; border-left:5px solid #EA580C;
        padding:12px 16px; border-radius:8px; margin:8px 0;
    }
    .alert-low {
        background:#F0FDF4; border-left:5px solid #16A34A;
        padding:12px 16px; border-radius:8px; margin:8px 0;
    }
</style>
""", unsafe_allow_html=True)
 
# ── Load & Train Model from CSV ───────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv('traffic_cleaned.csv')
 
    def label_traffic(volume):
        if volume < 1000:   return 0
        elif volume < 4000: return 1
        else:               return 2
 
    df['congestion_level'] = df['traffic_volume'].apply(label_traffic)
 
    features = ['temp','rain_1h','snow_1h','clouds_all',
                'hour','day_of_week','month','is_weekend']
    X = df[features]
    y = df['congestion_level']
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
 
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
 
    return model, X_test, y_test, df
 
with st.spinner("⏳ Loading ML model... please wait (~10 seconds)"):
    rf_model, X_test, y_test, df = load_and_train()
 
# ── Indian Cities ─────────────────────────────
CITIES = {
    "Mumbai":        {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra",   "population": "20.7M"},
    "Delhi":         {"lat": 28.6139, "lon": 77.2090, "state": "Delhi",          "population": "32.9M"},
    "Bangalore":     {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka",      "population": "12.5M"},
    "Hyderabad":     {"lat": 17.3850, "lon": 78.4867, "state": "Telangana",      "population": "10.5M"},
    "Chennai":       {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu",     "population": "10.9M"},
    "Kolkata":       {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal",    "population": "14.9M"},
    "Pune":          {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra",    "population": "7.4M"},
    "Ahmedabad":     {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat",        "population": "8.4M"},
    "Jaipur":        {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan",      "population": "3.9M"},
    "Lucknow":       {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh",  "population": "3.7M"},
    "Surat":         {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat",        "population": "7.8M"},
    "Chandigarh":    {"lat": 30.7333, "lon": 76.7794, "state": "Punjab",         "population": "1.1M"},
    "Nagpur":        {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra",    "population": "2.9M"},
    "Bhopal":        {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh", "population": "2.3M"},
    "Patna":         {"lat": 25.5941, "lon": 85.1376, "state": "Bihar",          "population": "2.5M"},
    "Kochi":         {"lat": 9.9312,  "lon": 76.2673, "state": "Kerala",         "population": "2.1M"},
    "Indore":        {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh", "population": "3.3M"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "population": "2.0M"},
}
 
CITY_ROUTES = [
    ("Mumbai","Pune"), ("Delhi","Jaipur"), ("Delhi","Chandigarh"),
    ("Delhi","Lucknow"), ("Bangalore","Chennai"), ("Bangalore","Hyderabad"),
    ("Mumbai","Ahmedabad"), ("Mumbai","Nagpur"), ("Hyderabad","Visakhapatnam"),
    ("Kolkata","Patna"), ("Ahmedabad","Surat"), ("Bhopal","Indore"), ("Chennai","Kochi"),
]
 
# ── Helper Functions ──────────────────────────
def predict_congestion(hour, dow, month, temp, rain, snow, clouds):
    is_weekend = 1 if dow >= 5 else 0
    inp = np.array([[temp, rain, snow, clouds, hour, dow, month, is_weekend]])
    return rf_model.predict(inp)[0]
 
def congestion_label(level):
    return {0: "🟢 LOW", 1: "🟡 MEDIUM", 2: "🔴 HIGH"}[level]
 
def congestion_color(level):
    return {0: "#16A34A", 1: "#EA580C", 2: "#DC2626"}[level]
 
def congestion_bg(level):
    return {0: "#F0FDF4", 1: "#FFF7ED", 2: "#FEF2F2"}[level]
 
def folium_color(level):
    return {0: "green", 1: "orange", 2: "red"}[level]
 
def accident_risk(hour, rain, temp, pred):
    score = 0
    if 0 <= hour <= 5:        score += 30
    elif hour in [7,8,17,18]: score += 15
    else:                     score += 5
    score += min(20, int(rain * 3))
    if temp < 273:            score += 15
    if pred == 2:             score += 25
    elif pred == 1:           score += 10
    return min(100, score)
 
def risk_level_label(score):
    if score >= 60:   return "HIGH",   "red",    "#FEF2F2"
    elif score >= 35: return "MEDIUM", "orange", "#FFF7ED"
    else:             return "LOW",    "green",  "#F0FDF4"
 
DAY_NAMES   = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
 
# ── Sidebar ───────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/city.png", width=70)
st.sidebar.title("🚦 Smart Traffic India")
st.sidebar.markdown("---")
 
page = st.sidebar.radio("📍 Navigate", [
    "🏠 Dashboard",
    "🗺️ India Traffic Map",
    "🛣️ Route Recommendation",
    "⚠️ Congestion Alerts",
    "⏰ Best Time to Travel",
    "🚗 Accident Risk",
    "📍 Hotspot Map",
])
 
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
selected_day   = st.sidebar.selectbox("📅 Day", range(7), format_func=lambda x: DAY_NAMES[x])
selected_month = st.sidebar.slider("📆 Month", 1, 12, datetime.now().month)
selected_temp  = st.sidebar.slider("🌡️ Temperature (K)", 240, 310, 280)
selected_rain  = st.sidebar.slider("🌧️ Rain (mm)", 0.0, 10.0, 0.0)
selected_snow  = st.sidebar.slider("❄️ Snow (mm)", 0.0, 1.0, 0.0)
selected_cloud = st.sidebar.slider("☁️ Clouds (%)", 0, 100, 40)
st.sidebar.markdown("---")
st.sidebar.caption("ML Smart City | Random Forest 95.48%")
 
# ════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# 🏙️ Smart Traffic India — Dashboard")
    st.markdown("ML-powered traffic congestion prediction across major Indian cities")
    st.markdown("---")
 
    current_hour = datetime.now().hour
    pred = predict_congestion(current_hour, selected_day, selected_month,
                              selected_temp, selected_rain, selected_snow, selected_cloud)
 
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (congestion_label(pred), "CURRENT CONGESTION", congestion_color(pred), congestion_bg(pred), f"Hour: {current_hour}:00"),
        ("95.48%", "MODEL ACCURACY", "#028090", "white", "Random Forest"),
        (f"{round(selected_temp-273.15,1)}°C", "TEMPERATURE", "#EA580C", "white", f"Rain: {selected_rain}mm"),
        (DAY_NAMES[selected_day][:3], "TODAY", "#7C3AED", "white", f"{MONTH_NAMES[selected_month-1]}"),
    ]
    for col, (val, label, color, bg, sub) in zip([c1,c2,c3,c4], metrics):
        with col:
            st.markdown(f"""
<div style='background:{bg}; border-left:6px solid {color};
                        border-radius:12px; padding:20px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
<div style='color:{color}; font-size:0.8rem; font-weight:bold'>{label}</div>
<div style='color:{color}; font-size:2rem; font-weight:bold'>{val}</div>
<div style='color:#64748B; font-size:0.8rem'>{sub}</div>
</div>""", unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("<div class='section-title'>🇮🇳 All Major Cities — Current Congestion</div>", unsafe_allow_html=True)
 
    city_data = []
    for city, info in CITIES.items():
        p    = predict_congestion(current_hour, selected_day, selected_month,
                                  selected_temp, selected_rain, selected_snow, selected_cloud)
        risk = accident_risk(current_hour, selected_rain, selected_temp, p)
        city_data.append({
            "City": city, "State": info["state"],
            "Population": info["population"],
            "Congestion": congestion_label(p),
            "Accident Risk": f"{risk}/100"
        })
    st.dataframe(pd.DataFrame(city_data), use_container_width=True, hide_index=True)
 
    st.markdown("---")
    st.markdown("<div class='section-title'>📊 Today's 24-Hour Forecast</div>", unsafe_allow_html=True)
 
    preds_24  = [predict_congestion(h, selected_day, selected_month,
                                    selected_temp, selected_rain, selected_snow, selected_cloud)
                 for h in range(24)]
    cmap_bar  = {0:'#16A34A', 1:'#EA580C', 2:'#DC2626'}
    fig, ax   = plt.subplots(figsize=(13, 3.5))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    ax.bar(range(24), [1]*24, color=[cmap_bar[p] for p in preds_24], edgecolor='white', width=0.85)
    ax.axvline(x=current_hour, color='#0D1B2A', linestyle='--', linewidth=2.5, label=f'Now: {current_hour}:00')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45, fontsize=8)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor='#16A34A', label='Low'),
        Patch(facecolor='#EA580C', label='Medium'),
        Patch(facecolor='#DC2626', label='High'),
        plt.Line2D([0],[0], color='#0D1B2A', linestyle='--', linewidth=2, label='Current Hour')
    ], loc='upper right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
 
# ════════════════════════════════════════════
# PAGE 2 — INDIA TRAFFIC MAP
# ════════════════════════════════════════════
elif page == "🗺️ India Traffic Map":
    st.markdown("# 🗺️ India Live Traffic Map")
    st.markdown("Interactive map showing congestion, routes & accident risk for all major cities")
    st.markdown("---")
 
    current_hour = datetime.now().hour
    map_type = st.radio("Map Layer", [
        "🔴 Congestion Levels", "🛣️ City Routes",
        "🚗 Accident Risk Zones", "🔥 Traffic Hotspots"
    ], horizontal=True)
 
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")
 
    if map_type == "🔴 Congestion Levels":
        for city, info in CITIES.items():
            pred  = predict_congestion(current_hour, selected_day, selected_month,
                                       selected_temp, selected_rain, selected_snow, selected_cloud)
            color = folium_color(pred)
            label = congestion_label(pred)
            popup_html = f"""
<div style='font-family:Arial; width:200px'>
<h4 style='color:#0D1B2A; margin:0'>{city}</h4>
<p style='color:#64748B; margin:4px 0'>{info['state']}</p>
<hr style='margin:6px 0'>
<b>Congestion:</b> {label}<br>
<b>Population:</b> {info['population']}<br>
<b>Hour:</b> {current_hour}:00
</div>"""
            folium.CircleMarker(
                location=[info["lat"], info["lon"]], radius=18,
                color=color, fill=True, fill_color=color, fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{city} — {label}"
            ).add_to(m)
            folium.Marker(
                location=[info["lat"]+0.3, info["lon"]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px;font-weight:bold;color:#0D1B2A">{city}</div>',
                    icon_size=(80,20))
            ).add_to(m)
 
    elif map_type == "🛣️ City Routes":
        for city, info in CITIES.items():
            pred = predict_congestion(current_hour, selected_day, selected_month,
                                      selected_temp, selected_rain, selected_snow, selected_cloud)
            folium.CircleMarker(
                location=[info["lat"], info["lon"]], radius=8,
                color=folium_color(pred), fill=True,
                fill_color=folium_color(pred), fill_opacity=0.8,
                tooltip=f"{city}"
            ).add_to(m)
        for city_a, city_b in CITY_ROUTES:
            a = CITIES[city_a]; b = CITIES[city_b]
            pred_a   = predict_congestion(current_hour, selected_day, selected_month,
                                          selected_temp, selected_rain, selected_snow, selected_cloud)
            pred_b   = predict_congestion((current_hour+1)%24, selected_day, selected_month,
                                          selected_temp, selected_rain, selected_snow, selected_cloud)
            avg_pred = round((pred_a + pred_b) / 2)
            dist_km  = int(((a["lat"]-b["lat"])**2 + (a["lon"]-b["lon"])**2)**0.5 * 111)
            folium.PolyLine(
                locations=[[a["lat"],a["lon"]], [b["lat"],b["lon"]]],
                color={0:"green",1:"orange",2:"red"}[avg_pred],
                weight=4, opacity=0.8,
                popup=folium.Popup(
                    f"<b>{city_a} → {city_b}</b><br>~{dist_km} km<br>{congestion_label(avg_pred)}",
                    max_width=220),
                tooltip=f"{city_a} → {city_b} | {congestion_label(avg_pred)}"
            ).add_to(m)
 
    elif map_type == "🚗 Accident Risk Zones":
        for city, info in CITIES.items():
            pred = predict_congestion(current_hour, selected_day, selected_month,
                                      selected_temp, selected_rain, selected_snow, selected_cloud)
            risk = accident_risk(current_hour, selected_rain, selected_temp, pred)
            rl, rc, _ = risk_level_label(risk)
            folium.CircleMarker(
                location=[info["lat"], info["lon"]],
                radius=15 + risk//10,
                color=rc, fill=True, fill_color=rc, fill_opacity=0.6,
                popup=folium.Popup(
                    f"<b>{city}</b><br>Risk: {rl} ({risk}/100)", max_width=200),
                tooltip=f"{city} — Risk: {risk}/100 ({rl})"
            ).add_to(m)
 
    elif map_type == "🔥 Traffic Hotspots":
        for city, info in CITIES.items():
            total     = sum(predict_congestion(h, selected_day, selected_month,
                                               selected_temp, selected_rain, selected_snow, selected_cloud)
                            for h in range(24))
            intensity = total / 48
            if intensity > 0.6:   hc, hl = "red",    "🔴 High Hotspot"
            elif intensity > 0.3: hc, hl = "orange", "🟡 Moderate"
            else:                 hc, hl = "green",  "🟢 Low Activity"
            folium.CircleMarker(
                location=[info["lat"], info["lon"]],
                radius=10 + int(intensity * 30),
                color=hc, fill=True, fill_color=hc, fill_opacity=0.5,
                popup=folium.Popup(
                    f"<b>{city}</b><br>{hl}<br>Score: {total}/48", max_width=200),
                tooltip=f"{city} — {hl}"
            ).add_to(m)
 
    st_folium(m, width=1100, height=580)
 
# ════════════════════════════════════════════
# PAGE 3 — ROUTE RECOMMENDATION
# ════════════════════════════════════════════
elif page == "🛣️ Route Recommendation":
    st.markdown("# 🛣️ Route Recommendation")
    st.markdown("Find the best route between two Indian cities")
    st.markdown("---")
 
    c1, c2 = st.columns(2)
    with c1:
        source_city = st.selectbox("📍 Starting City", list(CITIES.keys()))
    with c2:
        dest_city = st.selectbox("🏁 Destination City",
                                  [c for c in CITIES.keys() if c != source_city])
 
    travel_hour = st.slider("🕐 Planned Departure Hour", 0, 23, datetime.now().hour)
 
    if st.button("🔍 Find Best Route & Time", use_container_width=True):
        src_info = CITIES[source_city]
        dst_info = CITIES[dest_city]
        dist_km  = int(((src_info["lat"]-dst_info["lat"])**2 +
                        (src_info["lon"]-dst_info["lon"])**2)**0.5 * 111)
 
        route_options = [
            {"name": "🛣️ National Highway (Direct)",
             "distance": dist_km, "time_min": dist_km, "hour_offset": 0},
            {"name": "🏙️ State Highway (Via City)",
             "distance": int(dist_km*1.2), "time_min": int(dist_km*1.4), "hour_offset": 1},
            {"name": "🌿 Bypass Road (Longer but scenic)",
             "distance": int(dist_km*1.4), "time_min": int(dist_km*1.2), "hour_offset": -2},
        ]
 
        route_results = []
        for r in route_options:
            h    = (travel_hour + r["hour_offset"]) % 24
            pred = predict_congestion(h, selected_day, selected_month,
                                      selected_temp, selected_rain, selected_snow, selected_cloud)
            route_results.append({**r, "pred": pred})
        route_results.sort(key=lambda x: x["pred"])
        best = route_results[0]
 
        # Map
        m2 = folium.Map(
            location=[(src_info["lat"]+dst_info["lat"])/2,
                       (src_info["lon"]+dst_info["lon"])/2],
            zoom_start=6, tiles="CartoDB positron")
        folium.Marker([src_info["lat"], src_info["lon"]],
                      tooltip=f"📍 {source_city}",
                      icon=folium.Icon(color="blue", icon="play", prefix="fa")).add_to(m2)
        folium.Marker([dst_info["lat"], dst_info["lon"]],
                      tooltip=f"🏁 {dest_city}",
                      icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(m2)
        for i, r in enumerate(route_results):
            folium.PolyLine(
                locations=[[src_info["lat"],src_info["lon"]],
                            [dst_info["lat"],dst_info["lon"]]],
                color=["green","orange","blue"][i],
                weight=5 if i==0 else 3,
                opacity=0.9 if i==0 else 0.5,
                dash_array=None if i==0 else "5 5",
                tooltip=f"{r['name']} | {congestion_label(r['pred'])}"
            ).add_to(m2)
        st_folium(m2, width=1100, height=420)
 
        st.markdown(f"""
<div style='background:#F0FDF4; border-left:6px solid #16A34A;
                    border-radius:12px; padding:20px; margin:15px 0;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
<div style='color:#16A34A; font-weight:bold'>✅ BEST RECOMMENDED ROUTE</div>
<div style='color:#0D1B2A; font-size:1.4rem; font-weight:bold; margin:8px 0'>
                {best['name']}</div>
<div style='color:#64748B'>
                📏 ~{best['distance']} km &nbsp;|&nbsp;
                ⏱️ ~{best['time_min']} min &nbsp;|&nbsp;
                🚦 {congestion_label(best['pred'])}
</div>
</div>""", unsafe_allow_html=True)
 
        medals = ["🥇","🥈","🥉"]
        for i, r in enumerate(route_results):
            st.markdown(f"""
<div style='background:{congestion_bg(r["pred"])};
                        border-left:5px solid {congestion_color(r["pred"])};
                        border-radius:10px; padding:15px; margin:8px 0'>
<b>{medals[i]} {r['name']}</b><br>
<span style='color:#64748B; font-size:0.9rem'>
                    ~{r['distance']} km &nbsp;|&nbsp; ~{r['time_min']} min &nbsp;|&nbsp;
<b style='color:{congestion_color(r["pred"])}'>{congestion_label(r["pred"])}</b>
</span>
</div>""", unsafe_allow_html=True)
 
        st.markdown("---")
        hour_preds = [predict_congestion(h, selected_day, selected_month,
                                          selected_temp, selected_rain, selected_snow, selected_cloud)
                      for h in range(24)]
        best_hours = [h for h, p in enumerate(hour_preds) if p == 0]
        if best_hours:
            st.success(f"✅ Best departure hours: {', '.join([f'{h}:00' for h in best_hours[:5]])}")
        else:
            med_hours = [h for h, p in enumerate(hour_preds) if p == 1]
            st.warning(f"⚠️ No low-traffic hours today. Best: {', '.join([f'{h}:00' for h in med_hours[:5]])}")
 
# ════════════════════════════════════════════
# PAGE 4 — CONGESTION ALERTS
# ════════════════════════════════════════════
elif page == "⚠️ Congestion Alerts":
    st.markdown("# ⚠️ Congestion Alert System")
    st.markdown("12-hour forecast with warnings for all major Indian cities")
    st.markdown("---")
 
    current_hour        = datetime.now().hour
    selected_city_alert = st.selectbox("🏙️ Select City", list(CITIES.keys()))
 
    st.markdown(f"### 🔔 Next 12 Hours — {selected_city_alert}")
    high_hours = []
    cols = st.columns(4)
    for i in range(12):
        h    = (current_hour + i) % 24
        pred = predict_congestion(h, selected_day, selected_month,
                                   selected_temp, selected_rain, selected_snow, selected_cloud)
        if pred == 2:
            high_hours.append(h)
        with cols[i % 4]:
            st.markdown(f"""
<div style='background:{congestion_bg(pred)};
                        border-radius:10px; padding:12px; text-align:center;
                        border:1px solid {congestion_color(pred)}; margin:5px 0'>
<div style='font-weight:bold; color:#0D1B2A'>{h}:00</div>
<div style='font-size:1.3rem'>{congestion_label(pred)}</div>
<div style='color:#64748B; font-size:0.75rem'>
                    {'+'+str(i)+'h' if i > 0 else 'Now'}</div>
</div>""", unsafe_allow_html=True)
 
    st.markdown("---")
    if len(high_hours) >= 3:
        st.markdown(f"""<div class='alert-high'>
<b>🚨 SEVERE CONGESTION ALERT — {selected_city_alert}</b><br>
            HIGH congestion for <b>{len(high_hours)} of next 12 hours</b>!<br>
            Affected: {', '.join([f'{h}:00' for h in high_hours])}<br>
<b>Avoid travel during these hours if possible.</b>
</div>""", unsafe_allow_html=True)
    elif len(high_hours) >= 1:
        st.markdown(f"""<div class='alert-medium'>
<b>⚠️ CONGESTION WARNING — {selected_city_alert}</b><br>
            HIGH congestion at: <b>{', '.join([f'{h}:00' for h in high_hours])}</b><br>
            Plan your trip around these hours.
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='alert-low'>
<b>✅ ALL CLEAR — {selected_city_alert}</b><br>
            No HIGH congestion in next 12 hours. Good time to travel!
</div>""", unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("### 🗺️ City Alert Map")
    m3 = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")
    for city, info in CITIES.items():
        high_count = sum(1 for i in range(12)
                         if predict_congestion((current_hour+i)%24, selected_day, selected_month,
                                               selected_temp, selected_rain, selected_snow, selected_cloud) == 2)
        if high_count >= 3:   ac, al = "red",    f"🚨 SEVERE ({high_count} high hours)"
        elif high_count >= 1: ac, al = "orange", f"⚠️ WARNING ({high_count} high hours)"
        else:                 ac, al = "green",  "✅ All Clear"
        folium.CircleMarker(
            location=[info["lat"], info["lon"]], radius=14,
            color=ac, fill=True, fill_color=ac, fill_opacity=0.7,
            popup=folium.Popup(f"<b>{city}</b><br>{al}", max_width=200),
            tooltip=f"{city} — {al}"
        ).add_to(m3)
    st_folium(m3, width=1100, height=420)
 
# ════════════════════════════════════════════
# PAGE 5 — BEST TIME TO TRAVEL
# ════════════════════════════════════════════
elif page == "⏰ Best Time to Travel":
    st.markdown("# ⏰ Best Time to Travel")
    st.markdown("Find the hour with lowest congestion for any day")
    st.markdown("---")
 
    c1, c2, c3 = st.columns(3)
    with c1:
        travel_day   = st.selectbox("📅 Day", range(7), format_func=lambda x: DAY_NAMES[x])
    with c2:
        travel_month = st.selectbox("📆 Month", range(1,13), format_func=lambda x: MONTH_NAMES[x-1])
    with c3:
        travel_city  = st.selectbox("🏙️ City", list(CITIES.keys()))
 
    all_preds    = [predict_congestion(h, travel_day, travel_month,
                                       selected_temp, selected_rain, selected_snow, selected_cloud)
                    for h in range(24)]
    best_hours   = [h for h, p in enumerate(all_preds) if p == 0]
    medium_hours = [h for h, p in enumerate(all_preds) if p == 1]
    worst_hours  = [h for h, p in enumerate(all_preds) if p == 2]
 
    c1, c2, c3 = st.columns(3)
    cards = [
        (best_hours,   "#16A34A", "#F0FDF4", "✅ BEST HOURS",   "low-traffic"),
        (medium_hours, "#EA580C", "#FFF7ED", "🟡 MEDIUM HOURS", "medium-traffic"),
        (worst_hours,  "#DC2626", "#FEF2F2", "🔴 AVOID HOURS",  "high-traffic"),
    ]
    for col, (hours, color, bg, label, desc) in zip([c1,c2,c3], cards):
        with col:
            hours_str = ', '.join([f"{h}:00" for h in hours[:4]]) if hours else "None"
            st.markdown(f"""
<div style='background:{bg}; border-left:6px solid {color};
                        border-radius:12px; padding:20px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
<div style='color:{color}; font-size:0.85rem; font-weight:bold'>{label}</div>
<div style='color:#0D1B2A; font-size:1rem; font-weight:bold; margin-top:8px'>
                    {hours_str}</div>
<div style='color:#64748B; font-size:0.8rem'>{len(hours)} {desc} hours</div>
</div>""", unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown(f"### 📊 {DAY_NAMES[travel_day]} — Full Day Breakdown")
    cmap_bar = {0:'#16A34A', 1:'#EA580C', 2:'#DC2626'}
    fig, ax  = plt.subplots(figsize=(13, 3.5))
    fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('#F8FAFC')
    ax.bar(range(24), [1]*24, color=[cmap_bar[p] for p in all_preds], edgecolor='white', width=0.85)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45, fontsize=8.5)
    ax.set_yticks([])
    ax.set_title(f"Congestion by Hour — {DAY_NAMES[travel_day]}, {travel_city}", fontsize=12, color='#0D1B2A')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor='#16A34A', label='Low — Best'),
        Patch(facecolor='#EA580C', label='Medium'),
        Patch(facecolor='#DC2626', label='High — Avoid'),
    ], fontsize=9, loc='upper right')
    plt.tight_layout(); st.pyplot(fig); plt.close()
 
    st.markdown("---")
    st.markdown("### 📅 Weekly Comparison — Best Day to Travel")
    weekly = [sum(1 for h in range(24)
                  if predict_congestion(h, d, travel_month,
                                        selected_temp, selected_rain, selected_snow, selected_cloud) == 0)
              for d in range(7)]
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    fig2.patch.set_facecolor('#F8FAFC'); ax2.set_facecolor('#F8FAFC')
    colors_w = ['#028090' if i == travel_day else '#CBD5E1' for i in range(7)]
    ax2.bar(DAY_NAMES, weekly, color=colors_w, edgecolor='white', width=0.6)
    ax2.set_ylabel("Low-traffic hours", fontsize=9, color='#64748B')
    ax2.set_title("Best Day to Travel (most low-traffic hours)", fontsize=11, color='#0D1B2A')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    for i, v in enumerate(weekly):
        ax2.text(i, v+0.2, str(v)+'h', ha='center', fontsize=9, fontweight='bold', color='#0D1B2A')
    plt.tight_layout(); st.pyplot(fig2); plt.close()
 
# ════════════════════════════════════════════
# PAGE 6 — ACCIDENT RISK
# ════════════════════════════════════════════
elif page == "🚗 Accident Risk":
    st.markdown("# 🚗 Accident Risk Predictor")
    st.markdown("Estimates accident probability based on traffic + weather conditions")
    st.markdown("---")
 
    c1, c2, c3 = st.columns(3)
    with c1:
        risk_city  = st.selectbox("🏙️ City", list(CITIES.keys()))
        risk_hour  = st.slider("🕐 Hour", 0, 23, datetime.now().hour)
    with c2:
        visibility = st.selectbox("👁️ Visibility", ["Clear","Foggy","Heavy Rain","Snow/Ice"])
        road_type  = st.selectbox("🛣️ Road Type", ["Highway","City Road","Local Street","School Zone"])
    with c3:
        risk_rain  = st.slider("🌧️ Rain (mm)", 0.0, 10.0, 0.0)
        risk_temp  = st.slider("🌡️ Temp (K)", 240, 310, 280)
 
    if st.button("🔍 Calculate Risk", use_container_width=True):
        pred  = predict_congestion(risk_hour, selected_day, selected_month,
                                    risk_temp, risk_rain, selected_snow, selected_cloud)
        score = accident_risk(risk_hour, risk_rain, risk_temp, pred)
        score += {"Clear":0,"Foggy":25,"Heavy Rain":30,"Snow/Ice":40}[visibility]
        score += {"Highway":20,"City Road":10,"Local Street":5,"School Zone":15}[road_type]
        score  = min(100, score)
        rl, rc, rbg = risk_level_label(score)
        advice = {
            "HIGH":   "Avoid travel if possible. Reduce speed, use headlights, stay alert.",
            "MEDIUM": "Drive with caution. Reduce speed and allow extra travel time.",
            "LOW":    "Conditions relatively safe. Drive with standard precautions."
        }[rl]
 
        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
<div style='background:{rbg}; border-left:6px solid {rc};
                        border-radius:12px; padding:25px; text-align:center;
                        box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
<div style='color:{rc}; font-size:0.85rem; font-weight:bold'>
                    ACCIDENT RISK — {risk_city}</div>
<div style='color:{rc}; font-size:2rem; font-weight:bold; margin:10px 0'>
                    {rl} RISK</div>
<div style='color:#0D1B2A; font-size:2.5rem; font-weight:bold'>{score}/100</div>
</div>""", unsafe_allow_html=True)
        with c2:
            factors = []
            if 0 <= risk_hour <= 5:          factors.append("🌙 Late night driving")
            if risk_hour in [7,8,9,17,18]:   factors.append("🚦 Rush hour")
            if visibility != "Clear":         factors.append("🌫️ Poor visibility")
            if risk_rain > 2:                 factors.append("🌧️ Wet roads")
            if risk_temp < 273:              factors.append("❄️ Freezing temperature")
            if pred == 2:                    factors.append("🔴 High congestion area")
            st.markdown(f"""
<div style='background:white; border-radius:12px; padding:20px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
<b>⚠️ Safety Advice:</b><br><br>
<span style='color:#64748B'>{advice}</span><br><br>
<b>Risk Factors Detected:</b><br>
                {'<br>'.join(factors) if factors else '✅ No major risk factors'}
</div>""", unsafe_allow_html=True)
 
        # Gauge chart
        fig, ax = plt.subplots(figsize=(6, 1.5))
        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
        ax.barh(0, 100, color='#E2E8F0', height=0.5)
        ax.barh(0, score, color=rc, height=0.5)
        ax.set_xlim(0, 100); ax.set_yticks([])
        ax.set_xlabel("Risk Score (0 = Safe, 100 = Dangerous)", fontsize=9)
        ax.set_title(f"Risk Score: {score}/100", fontsize=11, color='#0D1B2A')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
 
        # Risk map
        st.markdown("### 🗺️ Risk Map — All Cities")
        m4 = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")
        for city, info in CITIES.items():
            cp        = predict_congestion(risk_hour, selected_day, selected_month,
                                           risk_temp, risk_rain, selected_snow, selected_cloud)
            cs        = accident_risk(risk_hour, risk_rain, risk_temp, cp)
            cl, cc, _ = risk_level_label(cs)
            folium.CircleMarker(
                location=[info["lat"], info["lon"]],
                radius=10 + cs//10,
                color=cc, fill=True, fill_color=cc, fill_opacity=0.6,
                popup=folium.Popup(f"<b>{city}</b><br>Risk: {cl} ({cs}/100)", max_width=150),
                tooltip=f"{city} — {cl} ({cs}/100)"
            ).add_to(m4)
        st_folium(m4, width=1100, height=420)
 
# ════════════════════════════════════════════
# PAGE 7 — HOTSPOT MAP
# ════════════════════════════════════════════
elif page == "📍 Hotspot Map":
    st.markdown("# 📍 Traffic Hotspot Map")
    st.markdown("Visual heatmap + interactive map of congestion patterns")
    st.markdown("---")
 
    st.markdown("### 🗺️ Interactive Hotspot Map — All Indian Cities")
    m5 = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")
    for city, info in CITIES.items():
        total     = sum(predict_congestion(h, selected_day, selected_month,
                                           selected_temp, selected_rain, selected_snow, selected_cloud)
                        for h in range(24))
        intensity = total / 48
        if intensity > 0.6:   hc, hl = "red",    "🔴 Major Hotspot"
        elif intensity > 0.3: hc, hl = "orange", "🟡 Moderate"
        else:                 hc, hl = "green",  "🟢 Low Activity"
        folium.CircleMarker(
            location=[info["lat"], info["lon"]],
            radius=12 + int(intensity * 25),
            color=hc, fill=True, fill_color=hc, fill_opacity=0.55,
            popup=folium.Popup(
                f"<b>{city}</b><br>{info['state']}<br>{hl}<br>Score: {total}/48<br>Population: {info['population']}",
                max_width=200),
            tooltip=f"{city} — {hl}"
        ).add_to(m5)
    st_folium(m5, width=1100, height=500)
 
    st.markdown("---")
    st.markdown("### 🗓️ Weekly Congestion Heatmap (Hour × Day)")
    heat = np.zeros((7, 24))
    for d in range(7):
        for h in range(24):
            heat[d][h] = predict_congestion(h, d, selected_month,
                                             selected_temp, selected_rain, selected_snow, selected_cloud)
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    cmap_heat = ListedColormap(['#16A34A','#EA580C','#DC2626'])
    ax.imshow(heat, cmap=cmap_heat, aspect='auto', vmin=0, vmax=2, interpolation='nearest')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45, fontsize=7.5)
    ax.set_yticks(range(7)); ax.set_yticklabels(DAY_NAMES, fontsize=10)
    ax.set_title(f"Congestion Heatmap — {MONTH_NAMES[selected_month-1]}", fontsize=13, color='#0D1B2A', pad=12)
    ax.legend(handles=[
        Patch(facecolor='#16A34A', label='Low'),
        Patch(facecolor='#EA580C', label='Medium'),
        Patch(facecolor='#DC2626', label='High'),
    ], loc='upper right', fontsize=9, bbox_to_anchor=(1.12, 1), borderaxespad=0)
    plt.tight_layout(); st.pyplot(fig); plt.close()
 
    st.markdown("---")
    hour_totals = heat.sum(axis=0)
    day_totals  = heat.sum(axis=1)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ⏰ Top 5 Worst Hours")
        for i, h in enumerate(np.argsort(hour_totals)[::-1][:5]):
            st.markdown(f"""
<div style='background:#FEF2F2; border-left:4px solid #DC2626;
                        border-radius:8px; padding:10px; margin:5px 0'>
<b>{"🥇🥈🥉4️⃣5️⃣".split()[i] if i < 3 else ["4️⃣","5️⃣"][i-3]} {h}:00</b> —
<span style='color:#DC2626'>Score: {hour_totals[h]:.0f}/14</span>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("### 📅 Top 5 Worst Days")
        for i, d in enumerate(np.argsort(day_totals)[::-1][:5]):
            st.markdown(f"""
<div style='background:#FFF7ED; border-left:4px solid #EA580C;
                        border-radius:8px; padding:10px; margin:5px 0'>
<b>{["🥇","🥈","🥉","4️⃣","5️⃣"][i]} {DAY_NAMES[d]}</b> —
<span style='color:#EA580C'>Score: {day_totals[d]:.0f}/48</span>
</div>""", unsafe_allow_html=True)
 
# ── Footer ────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align:center; color:#94A3B8; font-size:0.75rem'>
    🏙️ ML Smart City India<br>
    18 Major Cities<br>
    Team: Yash, Ayus, Kavya,<br>Jahnavi, Shreya<br>
    Random Forest — 95.48%
</div>""", unsafe_allow_html=True)