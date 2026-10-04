import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return np.array([])
    try:
        return np.genfromtxt(filename, invalid_raise=False)
    except Exception as e:
        print(f"Ошибка при чтении {filename}: {e}")
        return np.array([])

def zoom_factory(ax, base_scale=1.15):
    def zoom_fun(event):
        if event.inaxes != ax:
            return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1.0

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])
        ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])
        ax.figure.canvas.draw_idle()

    fig = ax.get_figure()
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

field = load_data('vector_field.txt')
ellipses = load_data('ellipse_levels.txt')
trajectories = load_data('trajectories.txt')

fig, ax = plt.subplots(figsize=(12, 8))

if field.ndim == 2 and field.shape[1] == 4:
    x, y, dx, dy = field[:, 0], field[:, 1], field[:, 2], field[:, 3]
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    
    ax.quiver(x, y, dx / norm, dy / norm, color='black', alpha=0.5, 
               pivot='middle', width=0.0022, headwidth=3, headlength=4, scale=35)

if ellipses.ndim == 2 and ellipses.shape[1] == 3:
    unique_c = np.unique(ellipses[:, 0])
    cmap = plt.cm.get_cmap('plasma', len(unique_c))
    for i, c in enumerate(unique_c):
        data = ellipses[ellipses[:, 0] == c]
        ax.plot(data[:, 1], data[:, 2], '--', color=cmap(i), linewidth=2.5, label=f'$V(x) = {c}$')

if trajectories.ndim == 2 and trajectories.shape[1] == 3:
    first = True
    for traj_id in np.unique(trajectories[:, 0]):
        data = trajectories[trajectories[:, 0] == traj_id]
        label = 'Траектории $\dot{x}=Ax$' if first else ""
        ax.plot(data[:, 1], data[:, 2], color='royalblue', alpha=0.55, linewidth=1.1, label=label)
        first = False

ax.set_title('Фазовый портрет и линии уровня функции Ляпунова $V(x)=x^T P x$', fontsize=13)
ax.set_xlabel('$x_1$', fontsize=11)
ax.set_ylabel('$x_2$', fontsize=11)
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(True, linestyle=':', alpha=0.6)

ax.set_xlim(-4.0, 4.0)
ax.set_ylim(-4.0, 4.0)
ax.set_aspect('equal', adjustable='box')

ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
plt.tight_layout()

zoom_factory(ax)

plt.show()