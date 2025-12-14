# -----------------------------
# streamlit_drone_spiral.py
# -----------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(layout="wide")

# -----------------------------
# 1️⃣ User Inputs
# -----------------------------
st.sidebar.header("Drone Mission Parameters")

U = st.sidebar.slider("Airspeed (m/s)", 5.0, 25.0, 14.0, 0.5)
Tmax = st.sidebar.slider("Max flight time (s)", 600, 3600, 1800, 60)
t_delay = st.sidebar.slider("Takeoff delay (s)", 0, 300, 60, 10)

wind_x = st.sidebar.slider("Wind X (m/s)", -5.0, 5.0, 2.0, 0.1)
wind_y = st.sidebar.slider("Wind Y (m/s)", -5.0, 5.0, 0.0, 0.1)

current_x = st.sidebar.slider("Current X (m/s)", -2.0, 2.0, 0.5, 0.05)
current_y = st.sidebar.slider("Current Y (m/s)", -2.0, 2.0, 0.0, 0.05)

r0 = st.sidebar.slider("Spiral base radius r₀ (m)", 0, 50, 20, 1)
b = st.sidebar.slider("Spiral sweep per rev b (m/rad)", 10, 100, 30, 1)

H_x = st.sidebar.number_input("Home X (m)", value=-300.0)
H_y = st.sidebar.number_input("Home Y (m)", value=0.0)

C0_x = st.sidebar.number_input("Datum start X (m)", value=0.0)
C0_y = st.sidebar.number_input("Datum start Y (m)", value=0.0)

# -----------------------------
# 2️⃣ Spiral / path functions
# -----------------------------
def theta_derivative(theta, U, r, b):
    # simple placeholder: constant airspeed along spiral
    return U / np.sqrt(r**2 + b**2)

def compute_spiral(t_span, params):
    U, r0, b, v_c, C0 = params['U'], params['r0'], params['b'], params['v_c'], params['C0']
    def ode(t, y):
        x, y_pos, theta = y
        r = r0 + b*theta
        dtheta_dt = theta_derivative(theta, U, r, b)
        dr_dt = b*dtheta_dt
        dx_dt = dr_dt*np.cos(theta) - r*dtheta_dt*np.sin(theta) + v_c[0]
        dy_dt = dr_dt*np.sin(theta) + r*dtheta_dt*np.cos(theta) + v_c[1]
        return [dx_dt, dy_dt, dtheta_dt]
    
    y0 = [0, 0, 0]
    sol = solve_ivp(ode, t_span, y0, max_step=1.0)
    x = sol.y[0] + C0[0]
    y_pos = sol.y[1] + C0[1]
    return sol.t, x, y_pos

def compute_full_path(params):
    H = params['H']
    C0 = params['C0']
    U = params['U']
    v_w = params['v_w']
    v_c = params['v_c']
    Tmax = params['Tmax']
    t_delay = params['t_delay']

    # Outbound leg
    d_out = np.linalg.norm(C0 - H)
    T_out = d_out / U
    t_out = np.linspace(t_delay, t_delay+T_out, 50)
    x_out = H[0] + (C0[0]-H[0])*(t_out-t_delay)/T_out
    y_out = H[1] + (C0[1]-H[1])*(t_out-t_delay)/T_out

    # Spiral leg
    t_spiral_span = (t_out[-1], t_out[-1]+Tmax*0.6)
    t_spiral, x_spiral, y_spiral = compute_spiral(t_spiral_span, params)

    # Return leg
    T_back = np.linalg.norm(H - np.array([x_spiral[-1], y_spiral[-1]])) / U
    t_back = np.linspace(t_spiral[-1], t_spiral[-1]+T_back, 50)
    x_back = x_spiral[-1] + (H[0]-x_spiral[-1])*(t_back-t_spiral[-1])/T_back
    y_back = y_spiral[-1] + (H[1]-y_spiral[-1])*(t_back-t_spiral[-1])/T_back

    # Concatenate
    t_full = np.concatenate([t_out, t_spiral, t_back])
    x_full = np.concatenate([x_out, x_spiral, x_back])
    y_full = np.concatenate([y_out, y_spiral, y_back])

    # Index ranges for phases
    outbound_idx = slice(0, len(t_out))
    spiral_idx = slice(len(t_out), len(t_out)+len(t_spiral))
    return_idx = slice(len(t_out)+len(t_spiral), len(t_full))

    return t_full, x_full, y_full, outbound_idx, spiral_idx, return_idx

# -----------------------------
# 3️⃣ Precompute full path
# -----------------------------
params = {
    'U': U, 'r0': r0, 'b': b,
    'v_c': np.array([current_x, current_y]),
    'v_w': np.array([wind_x, wind_y]),
    'C0': np.array([C0_x, C0_y]),
    'H': np.array([H_x, H_y]),
    'Tmax': Tmax,
    't_delay': t_delay
}

t_full, x_full, y_full, outbound_idx, spiral_idx, return_idx = compute_full_path(params)

# -----------------------------
# 4️⃣ Time slider
# -----------------------------
t_now = st.slider("Time (s)", float(t_full[0]), float(t_full[-1]), float(t_full[0]), step=1.0)

i = np.searchsorted(t_full, t_now)

# -----------------------------
# 5️⃣ Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10,8))

# Already passed path
ax.plot(x_full[:i], y_full[:i], 'b', label='Traveled')

# Future path grayed
ax.plot(x_full[i:], y_full[i:], 'gray', linestyle='dotted', label='Future')

# Phases visualization
ax.plot(x_full[outbound_idx], y_full[outbound_idx], 'b--', label='Outbound')
ax.plot(x_full[spiral_idx], y_full[spiral_idx], 'b-', label='Spiral')
ax.plot(x_full[return_idx], y_full[return_idx], 'b:', label='Return')

# Drone marker
ax.plot(x_full[i], y_full[i], 'ro', markersize=8)

# Home & datum
ax.plot(H_x, H_y, 'ks', label='Home')
ax.plot(C0_x, C0_y, 'k*', label='Datum start')

ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title("Interactive Drone Mission")
st.pyplot(fig)
