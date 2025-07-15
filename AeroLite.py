# AeroLite

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Freestream:
    def __init__(self, gamma=1.4, R=287.05, inputs={}, angle_of_attack_rad=0):
        self.gamma = gamma
        self.R = R
        self.aoa_rad = angle_of_attack_rad

        self.p = inputs.get('p', None)
        self.T = inputs.get('T', None)
        self.rho = inputs.get('rho', None)
        self.Mach = inputs.get('Mach', None)
        self.V = inputs.get('V', None)

        self._infer_missing()

        # Derived quantities
        self.a = np.sqrt(self.gamma * self.R * self.T)
        self.q = 0.5 * self.rho * self.V**2
        self.V_inf = np.array([np.cos(self.aoa_rad), 0, np.sin(self.aoa_rad)])
        self.V_inf = self.V_inf / np.linalg.norm(self.V_inf)

    def _infer_missing(self):
        known = [k for k in [self.p, self.T, self.rho, self.Mach, self.V] if k is not None]
        if len(known) < 3:
            raise ValueError("Need at least 3 of {p, T, rho, Mach, V} to compute freestream")

        # Ideal gas: p = rho R T
        if self.p is not None and self.T is not None and self.rho is None:
            self.rho = self.p / (self.R * self.T)
        if self.rho is not None and self.T is not None and self.p is None:
            self.p = self.rho * self.R * self.T
        if self.p is not None and self.rho is not None and self.T is None:
            self.T = self.p / (self.rho * self.R)

        # Speed of sound
        if self.T is not None:
            a = np.sqrt(self.gamma * self.R * self.T)
        else:
            raise ValueError("Temperature must be computable to get Mach/V")

        # Mach/V relationship
        if self.Mach is not None and self.V is None:
            self.V = self.Mach * a
        if self.V is not None and self.Mach is None:
            self.Mach = self.V / a


class Geometry:
    def __init__(self, vertices, faces, name="unnamed"):
        """
        Parameters:
            vertices (ndarray): (N, 3) array of 3D coordinates
            faces (ndarray): (M, 3) or (M, 4) array of vertex indices per face
        """
        self.name = name
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)

        self.normals, self.areas, self.valid_faces = self._compute_normals_and_areas()
        self.faces = self.valid_faces  # drop degenerate faces

    def _compute_normals_and_areas(self):
        normals = []
        areas = []
        valid_faces = []

        def normal_and_area(v0, v1, v2):
            u = v1 - v0
            v = v2 - v0
            n = np.cross(u, v)
            area = 0.5 * np.linalg.norm(n)
            if area < 1e-12:
                return np.zeros(3), 0.0
            return n / np.linalg.norm(n), area

        for tri in self.faces:
            v0, v1, v2 = self.vertices[tri]
            n, a = normal_and_area(v0, v1, v2)
            if a > 0:
                normals.append(n)
                areas.append(a)
                valid_faces.append(tri)

        return np.array(normals), np.array(areas), np.array(valid_faces)

    def plot(self, show_normals=False, normal_length=0.2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        trisurf = ax.plot_trisurf(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2],
                                  triangles=self.faces, color='lightblue', edgecolor='k', alpha=0.9)

        if show_normals:
            centroids = np.mean(self.vertices[self.faces], axis=1)
            ax.quiver(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                      self.normals[:, 0], self.normals[:, 1], self.normals[:, 2],
                      length=normal_length, color='r')

        ax.set_title(f"Geometry: {self.name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.axis('equal')
        plt.tight_layout()
        plt.show()



def create_cone(num_x, num_theta, span, radius, name="Cone", plot=True):
    """
    Creates a triangulated mesh of a right circular cone and returns a Geometry object.

    Parameters:
        num_x (int): Number of axial (lengthwise) divisions
        num_theta (int): Number of circumferential divisions
        span (float): Length of the cone from tip to base
        radius (float): Base radius
        name (str): Name for the Geometry object
        plot (bool): Whether to visualize the cone

    Returns:
        Geometry: a Geometry object with vertices, faces, normals, and areas
    """
    # Cone tip
    vertices = [(0.0, 0.0, 0.0)]  # index 0 is tip

    # Build circular rings down the cone
    x = np.linspace(0, span, num_x)
    theta = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)

    for i in range(1, num_x):  # skip tip
        xi = x[i]
        ri = xi * (radius / span)
        for t in theta:
            y = ri * np.cos(t)
            z = ri * np.sin(t)
            vertices.append((xi, y, z))
    vertices = np.array(vertices)

    # Build triangular faces
    faces = []

    # Tip fan (connect tip to first ring)
    for i in range(num_theta):
        v1 = 1 + i
        v2 = 1 + ((i + 1) % num_theta)
        faces.append([0, v2, v1])

    # Body panels
    for j in range(1, num_x - 1):
        ring_start = 1 + (j - 1) * num_theta
        next_ring_start = ring_start + num_theta
        for i in range(num_theta):
            i1 = ring_start + i
            i2 = ring_start + (i + 1) % num_theta
            i3 = next_ring_start + i
            i4 = next_ring_start + (i + 1) % num_theta
            faces.append([i1, i2, i3])
            faces.append([i2, i4, i3])
    faces = np.array(faces)

    # Create Geometry object
    geom = Geometry(vertices=vertices, faces=faces, name=name)

    # Optional plot
    if plot:
        geom.plot(show_normals=True)

    return geom


def newtonian_aero(geometry, freestream, plot=True, modified=True):
    """
    Computes Newtonian aerodynamic forces on a triangulated surface.

    Parameters:
        geometry (dict): 'vertices', 'faces', 'normals', 'areas'
        freestream (dict): 'angle_of_attack' (rad), 'velocity' (m/s), 'density' (kg/m^3), 'temperature' (K), 'pressure' (Pa), 'mach_number', 'gamma'
        plot (bool): Whether to plot the pressure coefficient distribution
        modified (bool): Use modified Newtonian theory if True, else use standard Newtonian theory

    Returns:
        forces (ndarray): (M, 3) force vector on each face
        lift (float): Total lift force
        drag (float): Total drag force
    """
    vertices = geometry.vertices
    faces = geometry.faces
    normals = geometry.normals
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)  # Ensure unit normals
    areas = geometry.areas
    
    aoa = freestream.aoa_rad
    velocity = freestream.V
    density = freestream.rho
    temperature = freestream.T
    pressure = freestream.p
    Mach = freestream.Mach
    gamma = freestream.gamma

    # Freestream velocity vector
    V_inf = np.array([np.cos(aoa), 0, np.sin(aoa)])
    V_inf = V_inf / np.linalg.norm(V_inf)  # Normalize freestream vector

    # Dynamic pressure
    q_inf = 0.5 * density * velocity**2

    # Theta is the angle between the normal and the freestream velocity
    cos_theta = normals @ V_inf  # (M,) cosine angles
    # Check if panel is leeward facing
    cos_theta[cos_theta > 0] = 0  # Set leeward facing panels to zero
    if modified:
        # Modified Newtonian theory
        Cp_max = 2 / (gamma * Mach ** 2) * ((((gamma + 1)**2 * Mach**2) / (4*gamma*Mach**2 - 2*(gamma-1)))**(gamma/(gamma-1))*((1 - gamma + 2*gamma*Mach**2)/(gamma+1))-1)
        Cp = Cp_max * cos_theta**2 # Newton's sine squared law with modified coefficient from compressible flow relations
    else:
        Cp = 2 * cos_theta**2 # Newton's sine squared law

    # Compute force per face
    forces = (Cp * q_inf * areas)[:, None] * normals  # (M, 3)

    # Integrate lift and drag
    total_force = np.sum(forces, axis=0)
    drag = -total_force[0] * np.cos(aoa) - total_force[2] * np.sin(aoa)
    lift = -total_force[2] * np.cos(aoa) + total_force[0] * np.sin(aoa) # Note reverse logic because Cps point opposite to the direction of lift generation (Cps on bottom surface point down but generate positive upward lift)

    # Optional pressure plot
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        trisurf = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                  triangles=faces, cmap='viridis', edgecolor='none', alpha=0.95,
                                  antialiased=True)
        # Color by Cp per face
        trisurf.set_array(Cp)
        trisurf.autoscale()
        ax.set_title('Pressure Coefficient (Cp)')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.axis('equal')
        # Plot freestream vector
        ax.quiver(0, 1, 0, V_inf[0], V_inf[1], V_inf[2], length=1.0, color='r', label='Freestream Velocity')
        ax.quiver(0, -1, 0, V_inf[0], V_inf[1], V_inf[2], length=1.0, color='r')
        ax.legend()
        plt.colorbar(trisurf, label='Cp')
        plt.tight_layout()
        plt.show()

    return forces, lift, drag

def vortex_lattice_aero(geometry, freestream, plot=True):
    """
    Placeholder for vortex lattice method aerodynamic analysis.
    Currently not implemented.
    """
    raise NotImplementedError("Vortex lattice method is not yet implemented.")

def vortex_panel_aero(geometry, freestream, plot=True):
    """
    Placeholder for vortex panel method aerodynamic analysis.
    Currently not implemented.
    """
    vertices = geometry.vertices
    faces = geometry.faces
    normals = geometry.normals
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)  # Ensure unit normals
    areas = geometry.areas
    
    aoa = freestream.aoa_rad
    velocity = freestream.V
    density = freestream.rho
    temperature = freestream.T
    pressure = freestream.p
    Mach = freestream.Mach
    gamma = freestream.gamma

    

    raise NotImplementedError("Vortex panel method is not yet implemented.")

geom = create_cone(num_x=50, num_theta=40, span=7.5, radius=1.4, name="Cone", plot=True)
fs = Freestream(
    gamma=1.4,
    R=287.05,
    inputs={
        'Mach': 19.0,
        'T': 220.0,
        'p': 1000.0  # Pa
    },
    angle_of_attack_rad=np.deg2rad(2)
)
forces, lift, drag = newtonian_aero(geom, fs, plot=True, modified=True)
print("Lift:", lift, "N")
print("Drag:", drag, "N")
   