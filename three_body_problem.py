import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation
from IPython.display import HTML
import plotly.graph_objects as go

# reference: https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767

def three_body_ode(state, t):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state
    
    # Calculate distances between bodies
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    
    # Calculate accelerations
    dax1 = K1*m2*(x2-x1)/r12**3 + K1*m3*(x3-x1)/r13**3
    day1 = K1*m2*(y2-y1)/r12**3 + K1*m3*(y3-y1)/r13**3
    
    dax2 = K1*m1*(x1-x2)/r12**3 + K1*m3*(x3-x2)/r23**3
    day2 = K1*m1*(y1-y2)/r12**3 + K1*m3*(y3-y2)/r23**3
    
    dax3 = K1*m1*(x1-x3)/r13**3 + K1*m2*(x2-x3)/r23**3
    day3 = K1*m1*(y1-y3)/r13**3 + K1*m2*(y2-y3)/r23**3
    
    # calculate velocities
    dvx1 = K2 * vx1
    dvy1 = K2 * vy1

    dvx2 = K2 * vx2
    dvy2 = K2 * vy2

    dvx3 = K2 * vx3
    dvy3 = K2 * vy3

    return [dvx1, dvy1, dvx2, dvy2, dvx3, dvy3, dax1, day1, dax2, day2, dax3, day3]

# Define constants
G = 6.67408e-11  # Gravitational constant

#Reference quantities
m_nd=1.989e+30 #kg - mass of the sun
r_nd=5.326e+12 #m - distance between stars in Alpha Centauri
v_nd=30000 #m/s - relative velocity of earth around the sun
t_nd=79.91*365*24*3600*0.51 #s - orbital period of Alpha Centauri

#Net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

# Initial conditions
m1, m2, m3 = 1.1, 0.9, 1.0 #kg Masses of the three bodies
p1, p2, p3 = [-0.5, 0.5], [0.0, 0.0], [0.0, 2.0]
v1, v2, v3 = [0.0, 0.1], [-0.1, 0.0], [0.1, -0.1]
initial_state = p1 + p2 + p3 + v1 + v2 + v3

# Time array
t = np.linspace(0, 5, 500)

# Solve ODE
solution = odeint(three_body_ode, initial_state, t)

# Extract positions
x1, y1 = solution[:, 0], solution[:, 1]
x2, y2 = solution[:, 2], solution[:, 3]
x3, y3 = solution[:, 4], solution[:, 5]

# Create a figure for dynamic plotting
fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title('Three-Body Problem Simulation')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('equal')
ax.grid(True)

# Initialize lines for the bodies
line1, = ax.plot([], [], 'r-', lw=2, label='Body 1')
line2, = ax.plot([], [], 'g-', lw=2, label='Body 2')
line3, = ax.plot([], [], 'b-', lw=2, label='Body 3')
ax.legend()

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

def update(frame):
    line1.set_data(x1[:frame], y1[:frame])
    line2.set_data(x2[:frame], y2[:frame])
    line3.set_data(x3[:frame], y3[:frame])
    return line1, line2, line3

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=30)

# Display the animation in the notebook
HTML(ani.to_jshtml())
