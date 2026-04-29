import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def plot_BH(plot_ergosphere,plot_outer_horizon,plot_inner_horizon,plot_singularity):
    n_theta = 200
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)
    # Outer 
    b = M + np.sqrt(M**2 - a**2)
    # iner 
    c = M - np.sqrt(M**2 - a**2)
    # Ergosphere rho
    rho = M + np.sqrt(M**2 - (a**2) * (np.cos(theta)**2))
    def sph_to_cart(r, theta, phi):
        x = np.sqrt(r**2+a**2)*np.sin(theta)*np.cos(phi)
        y = np.sqrt(r**2+a**2)*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return x, y, z
    if plot_ergosphere:
        x,y,z= sph_to_cart(rho, theta, phi)
        ax.plot_surface(x,y,z, alpha=0.3, color='yellow', linewidth=0)
    if plot_outer_horizon:
        r= b
        x,y,z = sph_to_cart(r, theta, phi)
        ax.plot_surface(x,y,z, alpha=0.4, color='orange', linewidth=0)
    if plot_inner_horizon:
        r = c
        x,y,z  = sph_to_cart(r, theta, phi)
        ax.plot_surface(x,y,z, alpha=0.4, color='red', linewidth=0)
    if plot_singularity:
        phi_ring = np.linspace(0, 2*np.pi, 500)
        x=a * np.cos(phi_ring)
        y=a * np.sin(phi_ring)
        z=np.zeros_like(phi_ring)
        ax.plot(x, y, z, color='purple', linewidth=3)

def delta(r,a,M):
    return r**2 - 2*M*r + a**2

def rho2(r,theta,a):
    return r**2 + a**2 * np.cos(theta)**2

def A(r,a,E,J):
    return (r**2+a**2)*E - a*J

def B(theta, a, E, J):
    return J -a*E*np.sin(theta)**2

# H = 1/(2 rho^2) F
def F(r, theta, pr, ptheta, a, M, E, J):
    Sigma = rho2(r,theta,a)
    Delta = delta(r,a,M)
    A_val = A(r,a,E,J)
    B_val = B(theta,a,E,J)
    
    return Delta*pr**2 +ptheta**2 + (B_val**2)/(np.sin(theta)**2) -(A_val**2)/Delta

def func(lam, y, a, M, E, J):
    r, theta, phi, pr, ptheta = y
    
    rhosquare = rho2(r,theta,a)
    Delta = delta(r,a,M)
    sin = np.sin(theta)
    cos = np.cos(theta)

    A_val = A(r,a,E,J)
    B_val = B(theta,a,E,J)

   
    dr = (Delta / rhosquare) * pr
    dtheta = ptheta / rhosquare

    dphi = (1/rhosquare)*(B_val/(sin**2) + a*A_val/rhosquare)

    drho2_dr = 2*r
    drho2_dtheta = -2*a**2*cos*sin
    dDelta_dr = 2*r - 2*M
    dA_dr = 2*r*E
    dB_dtheta = -2*a*E*sin*cos

   
    F_val = Delta*pr**2+ ptheta**2+(B_val**2)/(sin**2) -(A_val**2)/Delta

    dpr = -(1/(2*rhosquare)) * (dDelta_dr * pr**2  - (2*A_val*dA_dr)/Delta + (A_val**2 * dDelta_dr)/(Delta**2)) + (F_val/(2*rhosquare**2))* drho2_dr 

    dptheta = -(1/(2*rhosquare))* ((2*B_val*dB_dtheta)/(sin**2) -(2*B_val**2*cos)/(sin**3))+(F_val/(2*rhosquare**2))*drho2_dtheta
    return np.array([dr, dtheta, dphi, dpr, dptheta])

def get_pr(r,theta,ptheta,a,M,E,J,mu):
    rhosquare = rho2(r,theta,a)
    Delta = delta(r,a,M)
    A_val = A(r,a,E,J)
    B_val = B(theta,a,E,J)
    pr_val = (A_val**2)/Delta- ptheta**2 -(B_val**2)/(np.sin(theta)**2) -mu**2 * rhosquare
    pr = np.sqrt(max(pr_val/Delta, 0))
    return pr


def solve_trajectory(a,M, r0,theta0,ptheta0,E,J,massive,inward):
    if massive:
        mu =1
    else:
        mu=0
    if inward:
        pr0= -1*get_pr(r0,theta0,ptheta0,a,M,E,J,mu)
    else:
        pr0= get_pr(r0,theta0,ptheta0,a,M,E,J,mu)

    y0 = np.array([r0,theta0, 0.0,  pr0, ptheta0], dtype=float)
    # 0 for phi


    sol = solve_ivp(
        lambda t, y: func(t, y, a, M, E, J),
        (0, 1000),
        y0,
        max_step=0.05,
        rtol=1e-9,
        atol=1e-12
    )

    def H(r,theta,pr,ptheta):
        return F(r,theta,pr,ptheta,a,M,E,J)/(2*rho2(r,theta,a))

    H_vals = []
    for i in range(len(sol.t)):
        r, theta, _, pr, ptheta = sol.y[:, i]
        H_vals.append(H(r, theta, pr, ptheta))

    return [sol.t,sol.y[0], sol.y[1], sol.y[2], H_vals ]
    # time, radius, theta, phi, hamiltonian

#plotter
def plot_trajectory(ax,r,theta,phi):
    # oblate spheroid to cartesian
    x = np.sqrt(r**2 + a**2)*np.sin(theta)*np.cos(phi)
    y = np.sqrt(r**2 + a**2)*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    # Plot trajectory
    ax.plot(x, y, z, linewidth=1.5)
    ax.scatter(x[0], y[0], z[0], color='green', label='start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', label='end')

    if mu==0:
        ax.set_title("Photon",fontsize=20)
    else:
        ax.set_title("Massive Particle",fontsize=20)
    ax.legend(fontsize = 15)

#BH param
a = 1
M = 1
#trajectory and particle param
r0=2
theta0=np.pi/2
ptheta0=0
E = 0.3
J = 3
mu = 1
inward = True

t,r,theta,phi,H = solve_trajectory(a,M,r0,theta0,ptheta0,E,J,mu,inward)


fig = plt.figure(2,figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plot_trajectory(ax,r,theta,phi)
plot_BH(True,True,True,True)

limit = 10*M
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

fig2,ax2 = plt.subplots()
ax2.plot(t,H)
ax2.set_ylim([-2,2])
ax2.set_xlabel(r'$\lambda$')
ax2.set_ylabel("H")
ax2.set_title("Hamiltonian Conservation",fontsize= 20)
ax2.grid()
plt.show()



