import numpy as np
from vispy import app, scene
from vispy.scene import visuals
import fd_openmp

N_side = 4
N = N_side**3
L = 6.0
sigma = 1.0     
epsilon = 1.0

v_max = 5
dt = 0.01

x = np.linspace(L/(2*N_side), L - L/(2*N_side), N_side)
X, Y, Z = np.meshgrid(x, x, x)
pos = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T.astype(np.float32)

vel = (np.random.rand(N, 3).astype(np.float32) - 0.5) * 2 * v_max
vel -= np.mean(vel, axis=0)

#Using pybind-11 cpp-file function
acc_flat = fd_openmp.compute_acc_fd(pos.flatten().tolist(), N, L, sigma, epsilon)
acc = np.array(acc_flat, dtype=np.float32).reshape(N, 3)

#Animation
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True, bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.distance = L * 3.5

markers = visuals.Markers(spherical=True)
markers.parent = view.scene
colors = np.random.rand(N, 4)

box_coords = np.array([
    [0,0,0], [L,0,0], [L,L,0], [0,L,0], [0,0,0],
    [0,0,L], [L,0,L], [L,L,L], [0,L,L], [0,0,L],
    [L,0,L], [L,0,0], [L,L,0], [L,L,L], [0,L,L], [0,L,0]
])
box = visuals.Line(pos=box_coords, color='white', parent=view.scene)

def update(event):
    global pos, vel, acc
    
    pos = (pos + vel * dt + 0.5 * acc * dt**2) % L
    
    new_acc_flat = fd_openmp.compute_acc_fd(pos.flatten().tolist(), N, L, sigma, epsilon)
    new_acc = np.array(new_acc_flat, dtype=np.float32).reshape(N, 3)
    
    vel = vel + 0.5 * (acc + new_acc) * dt
    acc = new_acc
    
    markers.set_data(pos=pos, face_color=colors, size=15, edge_width=0)
    canvas.update()

timer = app.Timer(interval='auto', connect=update, start=True)

if __name__ == '__main__':
    app.run()