import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from scipy.integrate import solve_ivp

# === Lotka-Volterra Model ===
def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# === Initial Parameters ===
init_params = {
    'alpha': 1.1,
    'beta': 0.4,
    'delta': 0.1,
    'gamma': 0.4
}

# === Initial Conditions (modifiable) ===
init_prey = 10
init_pred = 5
z0 = [init_prey, init_pred]

# === Time Setup ===
t_span = (0, 30)
t_eval = np.linspace(*t_span, 1000)

# === Solver Function ===
def solve_model(alpha, beta, delta, gamma, z0):
    sol = solve_ivp(lotka_volterra, t_span, z0,
                    args=(alpha, beta, delta, gamma),
                    t_eval=t_eval)
    return sol

# === Format ODE Equation ===
def format_equation(alpha, beta, delta, gamma):
    return (
        r"$\frac{{dx}}{{dt}} = {:.2f}x - {:.2f}xy$".format(alpha, beta) + "\n" +
        r"$\frac{{dy}}{{dt}} = {:.2f}xy - {:.2f}y$".format(delta, gamma)
    )

# === Dark Mode Setup ===
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.5)

# === Initial Plot ===
sol = solve_model(**init_params, z0=z0)
prey_line, = ax.plot(sol.t, sol.y[0], label='Prey (x)', color='#1f77b4')
pred_line, = ax.plot(sol.t, sol.y[1], label='Predator (y)', color='#ff7f0e')

ax.set_xlabel('Time', color='white')
ax.set_ylabel('Population', color='white')
ax.set_title('Lotka-Volterra Predator-Prey Model', color='white')
ax.legend(facecolor='#222222', edgecolor='white', labelcolor='white')
ax.grid(True, color='#444444')
ax.tick_params(colors='white')

# === Sliders ===
ax_alpha = plt.axes([0.25, 0.42, 0.65, 0.03], facecolor='#222222')
ax_beta  = plt.axes([0.25, 0.37, 0.65, 0.03], facecolor='#222222')
ax_delta = plt.axes([0.25, 0.32, 0.65, 0.03], facecolor='#222222')
ax_gamma = plt.axes([0.25, 0.27, 0.65, 0.03], facecolor='#222222')

slider_alpha = Slider(ax_alpha, 'Prey Birth Rate (α):', 0.01, 3.0, valinit=init_params['alpha'])
slider_beta  = Slider(ax_beta,  'Predation Rate (β):', 0.01, 1.0, valinit=init_params['beta'])
slider_delta = Slider(ax_delta, 'Predator Birth Rate (δ):', 0.01, 1.0, valinit=init_params['delta'])
slider_gamma = Slider(ax_gamma, 'Predator Death Rate (γ):', 0.01, 3.0, valinit=init_params['gamma'])

for slider_ax in [ax_alpha, ax_beta, ax_delta, ax_gamma]:
    slider_ax.tick_params(colors='white')
    for label in slider_ax.get_xticklabels() + slider_ax.get_yticklabels():
        label.set_color('white')
    for spine in slider_ax.spines.values():
        spine.set_color('white')

for slider in [slider_alpha, slider_beta, slider_delta, slider_gamma]:
    slider.label.set_color('white')
# === Equation Display ===
eq_ax = plt.axes([0.1, 0.15, 0.8, 0.08])
eq_ax.axis('off')
eq_text = eq_ax.text(0.5, 0.5, format_equation(**init_params),
                     ha='center', va='center', fontsize=13, color='white',
                     transform=eq_ax.transAxes)
# === Text Boxes for Initial Populations ===
text_prey_ax = plt.axes([0.25, 0.08, 0.15, 0.04])
text_pred_ax = plt.axes([0.55, 0.08, 0.15, 0.04])

text_prey = TextBox(text_prey_ax, 'Initial Prey (x₀):', initial=str(init_prey))
text_pred = TextBox(text_pred_ax, 'Initial Predator (y₀):', initial=str(init_pred))

for box in [text_prey, text_pred]:
    box.label.set_color('white')        # Keep label white
    box.text_disp.set_color('black')    # Make input text black
    box.ax.set_facecolor('#222222')     # Dark background
    for spine in box.ax.spines.values():  # Set box border color
        spine.set_color('white')

# === Global to hold initial conditions ===
current_z0 = [init_prey, init_pred]

# === Update Function ===
def update(val=None):
    try:
        # Update initial conditions from text boxes
        current_z0[0] = float(text_prey.text)
        current_z0[1] = float(text_pred.text)
    except ValueError:
        return  # Ignore invalid input

    alpha = slider_alpha.val
    beta  = slider_beta.val
    delta = slider_delta.val
    gamma = slider_gamma.val

    sol = solve_model(alpha, beta, delta, gamma, current_z0)

    prey_line.set_ydata(sol.y[0])
    pred_line.set_ydata(sol.y[1])

    max_pop = max(sol.y[0].max(), sol.y[1].max())
    ax.set_ylim(0, max_pop * 1.1)

    eq_text.set_text(format_equation(alpha, beta, delta, gamma))
    fig.canvas.draw_idle()

# === Connect Inputs ===
slider_alpha.on_changed(update)
slider_beta.on_changed(update)
slider_delta.on_changed(update)
slider_gamma.on_changed(update)
text_prey.on_submit(update)
text_pred.on_submit(update)

# === Show Plot ===
plt.show(block=True)