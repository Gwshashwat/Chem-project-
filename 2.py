import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp

# Dark mode style
plt.style.use('dark_background')

# Nondimensional Lotka-Volterra system
def lotka_volterra(t, z, epsilon):
    X, Y = z
    dXdt = X * (1 - Y)
    dYdt = epsilon * Y * (X - 1)
    return [dXdt, dYdt]

# Simulation function
def simulate_and_plot(epsilon, X0, Y0):
    t_span = (0, 50)
    t_eval = np.linspace(*t_span, 1000)
    sol = solve_ivp(lotka_volterra, t_span, [X0, Y0], args=(epsilon,), t_eval=t_eval)

    ax.clear()
    ax.plot(sol.y[0], sol.y[1], color='cyan', lw=2, label=f"Trajectory from IC: ({X0:.2f}, {Y0:.2f})")
    ax.plot(1, 1, 'ro', label='Equilibrium (1, 1)')

    # Equation display
    eq_text = (
        r"$\frac{dX}{d\tau} = X(1 - Y)$" + "\n" +
        r"$\frac{dY}{d\tau} = \varepsilon Y(X - 1)$"
    )
    ax.text(1.05, 1.8, eq_text, fontsize=12, color='white', verticalalignment='top')

    ax.set_title("Phase Portrait of Nondimensional Lotka-Volterra Model", fontsize=14)
    ax.set_xlabel("Prey (X)")
    ax.set_ylabel("Predator (Y)")
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    ax.legend(loc='lower right')
    fig.canvas.draw_idle()

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.25, bottom=0.35)  # space for sliders

# Initial values
initial_epsilon = 1.0
initial_X0 = 1.2
initial_Y0 = 0.8

# Plot the initial simulation
simulate_and_plot(initial_epsilon, initial_X0, initial_Y0)

# Sliders: [left, bottom, width, height]
ax_epsilon = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor='black')
ax_X0 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor='black')
ax_Y0 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='black')

slider_epsilon = Slider(ax_epsilon, r'$\varepsilon$', 0.1, 3.0, valinit=initial_epsilon, valstep=0.1, color='deepskyblue')
slider_X0 = Slider(ax_X0, r'$X_0$ (Initial Prey)', 0.1, 2.5, valinit=initial_X0, valstep=0.1, color='lime')
slider_Y0 = Slider(ax_Y0, r'$Y_0$ (Initial Predator)', 0.1, 2.5, valinit=initial_Y0, valstep=0.1, color='orange')

# Update function
def update(val):
    epsilon = slider_epsilon.val
    X0 = slider_X0.val
    Y0 = slider_Y0.val
    simulate_and_plot(epsilon, X0, Y0)

# Connect sliders to update function
slider_epsilon.on_changed(update)
slider_X0.on_changed(update)
slider_Y0.on_changed(update)

# Reset button
reset_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset', color='gray', hovercolor='lightgray')

def reset(event):
    slider_epsilon.reset()
    slider_X0.reset()
    slider_Y0.reset()

reset_button.on_clicked(reset)

plt.show()