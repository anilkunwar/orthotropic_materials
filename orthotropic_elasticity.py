import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_stiffness_matrix(E1, E2, E3, nu12, nu23, nu31):
    # Calculate the engineering constants
    G12 = E2 / (2 * (1 + nu12))
    G23 = E3 / (2 * (1 + nu23))
    G31 = E1 / (2 * (1 + nu31))

    # Initialize the 6x6 stiffness matrix
    stiffness_matrix = np.zeros((6, 6))

    # Fill the upper-left 3x3 block (extensional strains)
    stiffness_matrix[0, 0] = 1 / E1
    stiffness_matrix[0, 1] = -nu12 / E1
    stiffness_matrix[0, 2] = -nu31 / E1
    stiffness_matrix[1, 0] = -nu12 / E2
    stiffness_matrix[1, 1] = 1 / E2
    stiffness_matrix[1, 2] = -nu23 / E2
    stiffness_matrix[2, 0] = -nu31 / E3
    stiffness_matrix[2, 1] = -nu23 / E3
    stiffness_matrix[2, 2] = 1 / E3

    # Fill the lower-right 3x3 block (shear strains)
    stiffness_matrix[3, 3] = 1 / G23
    stiffness_matrix[4, 4] = 1 / G31
    stiffness_matrix[5, 5] = 1 / G12

    # Copy the upper-right block to the lower-left block
    stiffness_matrix[3, 0] = stiffness_matrix[0, 3]
    stiffness_matrix[4, 1] = stiffness_matrix[1, 4]
    stiffness_matrix[5, 2] = stiffness_matrix[2, 5]

    # Copy the lower-right block to the upper-left block
    stiffness_matrix[0, 3] = stiffness_matrix[3, 0]
    stiffness_matrix[1, 4] = stiffness_matrix[4, 1]
    stiffness_matrix[2, 5] = stiffness_matrix[5, 2]

    return stiffness_matrix

def generate_compliance_matrix(E1, E2, E3, nu12, nu23, nu31):
    stiffness_matrix = generate_stiffness_matrix(E1, E2, E3, nu12, nu23, nu31)
    compliance_matrix = np.linalg.inv(stiffness_matrix)
    return compliance_matrix

def plot_elastic_moduli(E1, E2, E3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    Theta, Phi = np.meshgrid(theta, phi)
    X = np.sin(Phi) * np.cos(Theta)
    Y = np.sin(Phi) * np.sin(Theta)
    Z = np.cos(Phi) * (E1 * X + E2 * Y + E3 * (1 - X - Y))
    cmap = 'viridis'
    norm = plt.Normalize(Z.min(), Z.max())
    ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Elastic Moduli')
    fig.colorbar(ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm), label='Young\'s Modulus')
    
    return fig
def plot_shear_modulus(G12, G23, G31):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    Theta, Phi = np.meshgrid(theta, phi)
    X = np.sin(Phi) * np.cos(Theta)
    Y = np.sin(Phi) * np.sin(Theta)
    Z = np.cos(Phi) * (G12 * X + G23 * Y + G31 * (1 - X - Y))
    cmap = 'plasma'
    norm = plt.Normalize(Z.min(), Z.max())
    ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Shear Modulus')
    fig.colorbar(ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm), label='Shear Modulus')
    
    return fig
    
def plot_poissons_ratio(nu12, nu23, nu31):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    Theta, Phi = np.meshgrid(theta, phi)
    X = np.sin(Phi) * np.cos(Theta)
    Y = np.sin(Phi) * np.sin(Theta)
    Z = np.cos(Phi) * (nu12 * X + nu23 * Y + nu31 * (1 - X - Y))
    cmap = 'RdYlBu'
    norm = plt.Normalize(Z.min(), Z.max())
    ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel("Poisson's Ratio")
    ax.set_title("Poisson's Ratios")
    fig.colorbar(ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm), label="Poisson's Ratio")
    
    return fig



# Streamlit app
st.title("Orthotropic Material Properties")

# Input for Young's moduli
st.subheader("Young's Moduli")
E1 = st.number_input("E1 (Pa)", min_value=0.0, value=2.1e11, format="%e", key="E1")
E2 = st.number_input("E2 (Pa)", min_value=0.0, value=1.0e11, format="%e", key="E2")
E3 = st.number_input("E3 (Pa)", min_value=0.0, value=5.0e10, format="%e", key="E3")

# Input for Poisson's ratios
st.subheader("Poisson's Ratios")
nu12 = st.number_input("nu12", min_value=-1.0, max_value=0.5, value=0.2, key="nu12")
nu23 = st.number_input("nu23", min_value=-1.0, max_value=0.5, value=0.3, key="nu23")
nu31 = st.number_input("nu31", min_value=-1.0, max_value=0.5, value=0.1, key="nu31")

# Input for Shear moduli
st.subheader("Shear Moduli")
G12 = E2 / (2 * (1 + nu12))
G23 = E3 / (2 * (1 + nu23))
G31 = E1 / (2 * (1 + nu31))

if st.button("Generate Compliance Matrix"):
    compliance_matrix = generate_compliance_matrix(E1, E2, E3, nu12, nu23, nu31)
    compliance_matrix_exp = np.zeros_like(compliance_matrix, dtype=object)
    
    for i in range(compliance_matrix.shape[0]):
        for j in range(compliance_matrix.shape[1]):
            if compliance_matrix[i, j] != 0:
                compliance_matrix_exp[i, j] = "{:.2e}".format(compliance_matrix[i, j])
            else:
                compliance_matrix_exp[i, j] = "0"
    
    st.subheader("Compliance Matrix:")
    st.write(compliance_matrix_exp)

if st.button("Visualize Elastic Moduli"):
    fig_moduli = plot_elastic_moduli(E1, E2, E3)
    st.subheader("Elastic Moduli Visualization")
    st.pyplot(fig_moduli)


if st.button("Visualize Shear Modulus"):
    fig_shear = plot_shear_modulus(G12, G23, G31)
    st.subheader("Shear Modulus Visualization")
    st.pyplot(fig_shear)
    
if st.button("Visualize Poisson's Ratios"):
    fig_ratios = plot_poissons_ratio(nu12, nu23, nu31)
    st.subheader("Poisson's Ratios Visualization")
    st.pyplot(fig_ratios)
    

