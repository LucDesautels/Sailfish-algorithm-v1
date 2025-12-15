# streamlit_drone_spiral_animation.py

import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# -----------------------------
# 1️⃣ User Inputs
# -----------------------------
st.sidebar.header("Parameters")

U = st.sidebar.slider("Airspeed (m/s)", 5.0, 25.0, 14.0, 0.5)
Tmax = st.sidebar.slider("Max flight time (s)", 600, 3600, 1800, 60)
t_delay = st.sidebar.slider("Takeoff delay (s)", 0, 300, 60, 10)

wind_x = st.sidebar.slider("Wind X (m/s)", -5.0, 5.0, 2.0, 0.1)
wind_y = st.sidebar.slider("Wind Y (m/s)", -5.0, 5.0, 0.0, 0.1)

current_x = st.sidebar.slider("Current X (m/s)", -2.0, 2.0, 0.5, 0.05)
current_y = st.sidebar.slider("Current Y (m/s)", -2.0, 2.0, 0.4, 0.05)

r0 = st.sidebar.slider("Spiral base radius r₀ (m)", 0, 50, 20, 1)
b = st.sidebar.slider("Spiral sweep per rev b (m/rad)", 10, 100, 30, 1)

H_x = st.sidebar.number_input("Home X (m)", value=-300.0)
H_y = st.sidebar.number_input("Home Y (m)", value=15.0)

C0_x = st.sidebar.number_input("Datum start X (m)", value=10.0)
C0_y = st.sidebar.number_input("Datum start Y (m)", value=0.0)

# -----------------------------
# 2️⃣ Spiral / path functions
# -----------------------------
def theta_derivative(theta, U, r, b):
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
    v_c = params['v_c']
    Tmax = params['Tmax']
    t_delay = params['t_delay']
    r0 = params['r0']
    b = params['b']

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

    # Concatenate full path
    t_full = np.concatenate([t_out, t_spiral, t_back])
    x_full = np.concatenate([x_out, x_spiral, x_back])
    y_full = np.concatenate([y_out, y_spiral, y_back])
    
    C_x = C0_x + current_x * t_full
    C_y = C0_y + current_y * t_full


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
    'C0': np.array([C0_x, C0_y]),
    'H': np.array([H_x, H_y]),
    'Tmax': Tmax,
    't_delay': t_delay
}

t_full, x_full, y_full, outbound_idx, spiral_idx, return_idx = compute_full_path(params)

# -----------------------------
# 4️⃣ Create Plotly animation frames
# -----------------------------

#frames = []
#for i in range(len(t_full)):
#    # Past path gray, future path light gray
 #   frames.append(go.Frame(
  #      data=[
   #         go.Scatter(x=x_full[:i+1], y=y_full[:i+1], mode='lines+markers', line=dict(color='red'), marker=dict(size=4)),
    #        go.Scatter(x=[x_full[i]], y=[y_full[i]], mode='markers', marker=dict(color='red', size=12))
     #   ],
      #  name=str(i)
   # ))
frames = []
for i in range(len(t_full)):
    frames.append(go.Frame(
        data=[
            # Full past path of drone
            go.Scatter(
                x=x_full[:i+1],
                y=y_full[:i+1],
                mode='lines',
                line=dict(color='blue')
            ),

            # Drone position
            go.Scatter(
                x=[x_full[i]],
                y=[y_full[i]],
                mode='markers',
                marker=dict(color='red', size=12),
                name="Drone"
            ),

            # Datum position (MOVING)
            go.Scatter(
                x=[C_x[i]],
                y=[C_y[i]],
                mode='markers',
                marker=dict(color='green', size=10, symbol='x'),
                name="Datum"
            )
        ],
        name=str(i)
    ))

    
# -----------------------------
# 5️⃣ Create base figure
# -----------------------------
fig = go.Figure(
    data=[
        go.Scatter(x=x_full, y=y_full, mode='lines', line=dict(color='gray', dash='dot'), name='Full path'),
        go.Scatter(x=[H_x], y=[H_y], mode='markers', marker=dict(color='black', size=10, symbol='square'), name='Home'),
        go.Scatter(x=[C0_x], y=[C0_y], mode='markers', marker=dict(color='black', size=12, symbol='star'), name='Datum start')
    ],
    layout=go.Layout(
        title="Drone Spiral Animation",
        xaxis=dict(title="X (m)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (m)"),
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration":50, "redraw": True}, "fromcurrent": True, "mode": "immediate"}])])]
    ),
    frames=frames
)
fig.update_layout(
    xaxis=dict(
        showgrid=True,
        gridcolor="gray",
        gridwidth=1
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="gray",
        gridwidth=1
    )
)
st.plotly_chart(fig, width='stretch', height=800)
