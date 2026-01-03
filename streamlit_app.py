import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

st.set_page_config(layout="wide")
st.title("3D Function Explorer — surface + partial derivatives")

st.sidebar.header("Function input")
expr_input = st.sidebar.text_area("Enter f(x, y)", value="sin(x)*cos(y)", height=80)

st.sidebar.header("Plot settings")
x_min, x_max = st.sidebar.slider("x range", -10.0, 10.0, (-5.0, 5.0), step=0.5)
y_min, y_max = st.sidebar.slider("y range", -10.0, 10.0, (-5.0, 5.0), step=0.5)
resolution = st.sidebar.slider("Grid resolution", 20, 200, 100)

st.sidebar.header("Evaluate at a point")
px = st.sidebar.number_input("x", value=0.0, format="%.4f")
py = st.sidebar.number_input("y", value=0.0, format="%.4f")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Surface plot")

with col2:
    st.subheader("Expressions")

# Prepare symbols and parsing
x, y = sp.symbols("x y")
transformations = (standard_transformations + (implicit_multiplication_application,))
local_dict = {"x": x, "y": y}

try:
    expr = parse_expr(expr_input, local_dict=local_dict, transformations=transformations)
except Exception as e:
    st.error(f"Could not parse expression: {e}")
    st.stop()

# Compute partial derivatives
df_dx = sp.diff(expr, x)
df_dy = sp.diff(expr, y)

# Display expressions
with col2:
    st.markdown("**f(x, y)**")
    st.latex(sp.latex(expr))
    st.markdown("**∂f/∂x**")
    st.latex(sp.latex(df_dx))
    st.markdown("**∂f/∂y**")
    st.latex(sp.latex(df_dy))

# Numeric lambdified functions
try:
    f_num = sp.lambdify((x, y), expr, modules=["numpy"])
    dfdx_num = sp.lambdify((x, y), df_dx, modules=["numpy"])
    dfdy_num = sp.lambdify((x, y), df_dy, modules=["numpy"])
except Exception as e:
    st.error(f"Could not create numerical functions: {e}")
    st.stop()

# Build grid and evaluate
xs = np.linspace(x_min, x_max, resolution)
ys = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(xs, ys)
try:
    Z = f_num(X, Y)
    Z = np.array(Z, dtype=float)
except Exception as e:
    st.error(f"Error evaluating function on grid: {e}")
    st.stop()

Z = np.nan_to_num(Z, nan=np.nan, posinf=np.nan, neginf=np.nan)

fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)"), margin=dict(l=0, r=0, t=30, b=0))

with col1:
    st.plotly_chart(fig, use_container_width=True)

# Evaluate partial derivatives at chosen point
try:
    val_fx = float(dfdx_num(px, py))
    val_fy = float(dfdy_num(px, py))
    val_f = float(f_num(px, py))
except Exception:
    val_fx = None
    val_fy = None
    val_f = None

with st.expander("Evaluate at point"):
    st.write(f"Point: (x={px:.4f}, y={py:.4f})")
    if val_f is not None:
        st.write(f"f = {val_f:.6g}")
    else:
        st.write("f: could not evaluate at this point")
    if val_fx is not None:
        st.write(f"∂f/∂x = {val_fx:.6g}")
    else:
        st.write("∂f/∂x: could not evaluate at this point")
    if val_fy is not None:
        st.write(f"∂f/∂y = {val_fy:.6g}")
    else:
        st.write("∂f/∂y: could not evaluate at this point")

st.markdown("---")
st.caption("Supports common sympy functions: sin, cos, exp, log, sqrt, etc. Use 'x' and 'y' as variables.")
