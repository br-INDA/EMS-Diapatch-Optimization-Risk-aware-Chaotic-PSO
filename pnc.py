import streamlit as st
import pandas as pd
import math
import folium
from streamlit_folium import st_folium
import networkx as nx
import osmnx as ox
import traceback
import random
import time  
from branca.element import Template, MacroElement

# ================= Helper Functions ================= #

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def ensure_edge_lengths(G):
    try:
        return ox.distance.add_edge_lengths(G)
    except Exception:
        return G

def download_osm_bbox(north, south, east, west):
    try:
        G = ox.graph_from_bbox(
            north, south, east, west,
            network_type="drive_service",
            simplify=True,
            retain_all=True
        )
    except Exception:
        G = ox.graph_from_bbox(north, south, east, west,
                               "drive_service", True, True)
    return ensure_edge_lengths(G)

def nearest_node_safe(G, lat, lon):
    try:
        return ox.nearest_nodes(G, lon, lat)
    except Exception:
        return ox.distance.nearest_nodes(G, lon, lat)

def nodes_to_coords(G, route):
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

def route_length_km(G, route):
    total = 0
    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            continue
        key = next(iter(edge_data.keys()))
        total += edge_data[key].get("length", 0)
    return total / 1000.0

def compute_bbox(points, buffer_m=800):
    lats = [p[0] for p in points]
    lngs = [p[1] for p in points]
    delta = buffer_m / 111000.0
    return max(lats) + delta, min(lats) - delta, max(lngs) + delta, min(lngs) - delta

def risk_to_color(risk):
    if risk <= 3:
        return "lightgreen"
    elif risk <= 7:
        return "orange"
    else:
        return "red"

# ================= UI Setup ================= #

st.set_page_config(layout="wide", page_title="EMS Dispatch - Chaotic PSO")
st.title("🚑 EMS Dispatch System — Chaotic PSO Optimization")

file = st.file_uploader("Upload Hospital Locations (.csv or .xlsx)", ["csv", "xlsx"])
if not file:
    st.info("Please upload a file containing hospital latitudes and longitudes.")
    st.stop()

df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

name_col = st.selectbox("Hospital Name Column", df.columns)
lat_col  = st.selectbox("Latitude Column", df.columns)
lng_col  = st.selectbox("Longitude Column", df.columns)

df.dropna(subset=[lat_col, lng_col], inplace=True)
df.reset_index(drop=True, inplace=True)

# ================= Session State ================= #

if "emergencies" not in st.session_state:
    st.session_state["emergencies"] = []
if "pending" not in st.session_state:
    st.session_state["pending"] = None

# ✅ Persistent storage for results + runtime
if "pso_results" not in st.session_state:
    st.session_state["pso_results"] = None
if "algo_time" not in st.session_state:
    st.session_state["algo_time"] = None

extra_amb = st.slider("Extra Random Ambulances", 0, 20, 5)

col1, col2 = st.columns(2)
if col1.button("↩ Undo Last Emergency") and st.session_state["emergencies"]:
    st.session_state["emergencies"].pop()
    st.session_state["pso_results"] = None
    st.session_state["algo_time"] = None
    st.rerun()

if col2.button("🗑 Clear All"):
    st.session_state["emergencies"].clear()
    st.session_state["pending"] = None
    st.session_state["pso_results"] = None
    st.session_state["algo_time"] = None
    st.rerun()

# ================= Map Interaction ================= #

center = (df[lat_col].mean(), df[lng_col].mean())
m = folium.Map(location=center, zoom_start=13)

for _, r in df.iterrows():
    folium.Marker(
        [r[lat_col], r[lng_col]],
        tooltip=f"🏥 {r[name_col]}",
        icon=folium.Icon(color="green", prefix="glyphicon", icon="plus-sign")
    ).add_to(m)

for i, (lat, lng, risk) in enumerate(st.session_state["emergencies"], start=1):
    folium.Marker(
        [lat, lng],
        tooltip=f"Emergency {i} | Risk {risk}",
        icon=folium.Icon(color=risk_to_color(risk), prefix="glyphicon", icon="info-sign")
    ).add_to(m)

map_data = st_folium(m, height=500, width="100%")

if map_data and map_data.get("last_clicked"):
    st.session_state["pending"] = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])

if st.session_state["pending"]:
    with st.form("risk_form"):
        risk = st.slider("Set Risk Level for Clicked Point", 1, 10, 5)
        if st.form_submit_button("Add Emergency"):
            lat, lng = st.session_state["pending"]
            st.session_state["emergencies"].append((lat, lng, risk))
            st.session_state["pending"] = None
            st.session_state["pso_results"] = None
            st.session_state["algo_time"] = None
            st.rerun()

emer = st.session_state["emergencies"]
if not emer:
    st.warning("Click the map to place emergencies.")
    st.stop()

# ================= Chaotic PSO Algorithm ================= #

def pso_optimize_chaotic(ambs, hosps, ems, it=60, np=40):
    A, H, M = len(ambs), len(hosps), len(ems)
    dim = 2 * M

    def fitness(pos):
        total = 0.0
        used_ambs = [0] * A
        for j in range(M):
            ai = int(round(pos[j])) % A
            hi = int(round(pos[M + j])) % H
            used_ambs[ai] += 1

            ax, ay = ambs[ai]["coord"]
            ex, ey, risk = ems[j]
            hx, hy = hosps[hi]["coord"]

            d1 = haversine(ax, ay, ex, ey)
            d2 = haversine(ex, ey, hx, hy)

            score = (d1 + d2) * (1 + risk / 10.0)

            if d1 > 4 and risk > 7:
                score += 50
            total += score

        for count in used_ambs:
            if count > 1:
                total += (count - 1) * 500
        return total

    cx = random.random()
    mu = 4.0

    particles = [[random.uniform(0, max(A, H)) for _ in range(dim)] for _ in range(np)]
    vel = [[0.0] * dim for _ in range(np)]
    pbest = [p.copy() for p in particles]
    pscore = [fitness(p) for p in particles]
    gbest = pbest[pscore.index(min(pscore))].copy()

    for _ in range(it):
        cx = mu * cx * (1 - cx)
        w_chaotic = 0.4 + (0.5 * cx)

        for i in range(np):
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                vel[i][d] = (w_chaotic * vel[i][d] +
                             1.5 * r1 * (pbest[i][d] - particles[i][d]) +
                             1.5 * r2 * (gbest[d] - particles[i][d]))
                particles[i][d] += vel[i][d]

            f_new = fitness(particles[i])
            if f_new < pscore[i]:
                pscore[i] = f_new
                pbest[i] = particles[i].copy()

        gbest = pbest[pscore.index(min(pscore))].copy()

    return [(int(round(gbest[j])) % A, int(round(gbest[M + j])) % H) for j in range(M)]

# ================= Execution & Results ================= #

if st.button("🚨 Run Chaotic Optimization"):
    ambulances = [{"coord": (float(r[lat_col]), float(r[lng_col])), "source": "hospital"} for _, r in df.iterrows()]
    n_h, s_h, e_h, w_h = compute_bbox([a["coord"] for a in ambulances])
    for _ in range(extra_amb):
        ambulances.append({"coord": (random.uniform(s_h, n_h), random.uniform(w_h, e_h)), "source": "random"})

    hospitals = [{"coord": (float(df.loc[i, lat_col]), float(df.loc[i, lng_col])), "name": df.loc[i, name_col]} for i in range(len(df))]

    try:
        # ✅ TIME START
        start = time.perf_counter()

        assignments = pso_optimize_chaotic(ambulances, hospitals, emer)

        # ✅ TIME END
        end = time.perf_counter()
        st.session_state["algo_time"] = end - start

        report_rows = []
        for j, (ai, hi) in enumerate(assignments):
            ex, ey, risk = emer[j]
            ax, ay = ambulances[ai]["coord"]
            hx, hy = hospitals[hi]["coord"]
            dist = haversine(ax, ay, ex, ey) + haversine(ex, ey, hx, hy)

            report_rows.append({
                "Emergency": j + 1,
                "Risk": risk,
                "Hospital": hospitals[hi]['name'],
                "Total Est. Dist (km)": round(dist, 2),
                "Route_Coords": [(ax, ay), (ex, ey), (hx, hy)],
                "Risk_Color": risk_to_color(risk)
            })

        st.session_state["pso_results"] = {
            "report": pd.DataFrame(report_rows),
            "hospitals": hospitals,
            "ambulances": ambulances,
            "emer": emer
        }

    except Exception as e:
        st.error(f"Optimization Error: {e}")
        st.text(traceback.format_exc())

# ✅ Show results + runtime if they exist
if st.session_state.get("pso_results"):
    res = st.session_state["pso_results"]

    # ✅ Persistent runtime display
    if st.session_state.get("algo_time") is not None:
        st.info(f"⏱️ Chaotic PSO Runtime: {st.session_state['algo_time']:.4f} seconds")

    st.subheader("📋 Optimized Dispatch Schedule (Chaotic PSO)")
    st.table(res["report"][["Emergency", "Risk", "Hospital", "Total Est. Dist (km)"]])

    all_coords = [a["coord"] for a in res["ambulances"]] + [h["coord"] for h in res["hospitals"]]
    n, s, e, w = compute_bbox(all_coords)

    m_res = folium.Map()
    m_res.fit_bounds([[s, w], [n, e]])

    for h in res["hospitals"]:
        folium.Marker(h["coord"], tooltip=h["name"],
                      icon=folium.Icon(color="green", icon="plus-sign")).add_to(m_res)

    for a in res["ambulances"]:
        folium.Marker(a["coord"], tooltip="Ambulance",
                      icon=folium.Icon(color="blue", icon="truck", prefix="fa")).add_to(m_res)

    for _, row in res["report"].iterrows():
        coords = row["Route_Coords"]
        folium.PolyLine(coords[:2], color="red", weight=4, opacity=0.8).add_to(m_res)
        folium.PolyLine(coords[1:], color="green", weight=4, opacity=0.8).add_to(m_res)
        folium.Marker(coords[1], tooltip=f"Risk {row['Risk']}",
                      icon=folium.Icon(color=row["Risk_Color"])).add_to(m_res)

    st_folium(m_res, height=600, width="100%", key="results_map")
