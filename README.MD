# Astarte Dynamics with Square Mapping  
*An Integrated Framework for Mapping Helical Dynamics to Square Geometry*  

**Authors:**  
*Wormwood*  

**Affiliation:**  
*Humanity*  

**Date:**  
*February 2025 13, 12:09*

---

## Abstract

We introduce a novel framework—Astarte Dynamics with Square Mapping—that unifies circular (helical) dynamics with square geometry through the extraction of critical crossing events. In this framework, a dynamic system governed by relativistic energy‐transfer constraints evolves over time, and its key “number vector” events are identified by sign changes in the derivative of the null channel. These events are characterized by their phase, which we then remap from the unit circle to a square via an exact mathematical transformation. The resulting formulation enables precise measurement and analysis both from the “inside” and “outside” of the square, providing a new tool for dynamic pattern recognition and geometric transformations.

---

## 1. Introduction

Circular dynamics are ubiquitous in nature—from the helical structures of DNA to the rotational motion of galaxies. However, recent observations suggest that critical events in these dynamics can be mapped to square geometries. In the Astarte model, the interplay between base registers and a null (feedforward) channel produces helical dynamics subject to a relativistic energy‐transfer cap. When the derivative of the null channel changes sign, a crossing event occurs. These events form a “number vector” whose phase information, when remapped, reveals an intrinsic square structure. This paper details the full mathematical formulation of the Astarte dynamics, the extraction of crossing events, and the precise mapping from the circle to the square.

---

## 2. Mathematical Formulation

### 2.1 Astarte Dynamics

Let the following parameters be defined:

- **\(I\)**: Constant input.
- **\(\kappa, \lambda\)**: Scaling constants, with the scaled input \(A = \kappa\,\lambda\,I\).
- **\(\mu\)**: Blend factor for the base register update.
- **\(\omega\)**: Base angular frequency (radians per time unit).
- **\(\phi\)**: Phase offset.
- **\(\zeta\)**: Feedforward adjustment factor.
- **\(\delta\)**: Capacitive correction strength.
- **\(c\)**: Energy‐transfer cap (analogous to a relativistic limit).
- **\(\Delta t\)**: Small time increment.

Define the instantaneous phase:
\[
\theta(t) = \omega\,t + \phi.
\]

The base registers \(x_A(t)\) and \(x_B(t)\) evolve as:
\[
\begin{aligned}
x_A(t+\Delta t) &= X_0(t) + (1-\mu)\Bigl[x_A(t)-X_0(t)\Bigr] + \mu\,A\,\cos\bigl(\theta(t)\bigr),\\[1mm]
x_B(t+\Delta t) &= X_0(t) + (1-\mu)\Bigl[x_B(t)-X_0(t)\Bigr] + \mu\,A\,\sin\bigl(\theta(t)\bigr),
\end{aligned}
\]
with the difference
\[
\Delta(t) = x_A(t) - x_B(t).
\]

The null channel \(X_0(t)\) is updated according to:
\[
X_0(t+\Delta t) = X_0(t) + \left\{ \zeta\Bigl(|I| - |X_0(t)|\Bigr) - \delta\,\Delta(t)\,\sqrt{t}\,\right\}\,\sqrt{1-\frac{|X_0(t)|^2}{c^2}}.
\]
This update enforces the invariant \( |X_0(t)| \le c \).

---

### 2.2 Extraction of Crossing Events (The Number Vector)

The instantaneous frequency offset is approximated by:
\[
\frac{dX_0}{dt} \approx \frac{X_0(t+\Delta t)-X_0(t)}{\Delta t}.
\]
A **crossing event** occurs at time \(t_i\) when the sign of \(\frac{dX_0}{dt}\) changes. At each crossing event, we record:
- The time \(t_i\),
- The corresponding null channel value \(X_0(t_i)\),
- The phase \(\theta(t_i) = \omega\,t_i + \phi\).

The number vector is then given by:
\[
\{(t_i, X_0(t_i), \theta(t_i))\}_{i\in \mathbb{N}}.
\]

---

### 2.3 Mapping from the Circle to a Square

Each crossing event has an associated phase \(\theta(t_i)\). To map the point on the unit circle \((\cos\theta, \sin\theta)\) to a square, we apply the transformation:
\[
\begin{aligned}
x_s(t_i) &= \frac{\cos\theta(t_i)}{\max\{|\cos\theta(t_i)|,\,|\sin\theta(t_i)|\}},\\[1mm]
y_s(t_i) &= \frac{\sin\theta(t_i)}{\max\{|\cos\theta(t_i)|,\,|\sin\theta(t_i)|\}}.
\end{aligned}
\]
This mapping is exact and allows measurements to be made with full precision from either the inside or the outside of the square.

---

## 3. Simulation and Implementation

The complete implementation of the Astarte dynamics, extraction of crossing events, and square mapping is provided in the following Python script.

```python
#!/usr/bin/env python
"""
===========================================================================
Astarte Dynamics with Square Mapping – Whitepaper Release
===========================================================================
This file implements the full unified framework for Astarte dynamics,
extracting critical crossing events (the number vector) and mapping these
events from circular to square geometry.
===========================================================================
Mathematical Formulation:
1. Dynamics:
   θ(t) = ω*t + φ,   A = κ*λ*I,   Δ(t) = x_A(t) - x_B(t)
   x_A(t+dt) = X₀(t) + (1-μ)[x_A(t)-X₀(t)] + μ*A*cos(θ(t))
   x_B(t+dt) = X₀(t) + (1-μ)[x_B(t)-X₀(t)] + μ*A*sin(θ(t))
   X₀(t+dt) = X₀(t) + { ζ*(|I| - |X₀(t)|) - δ*Δ(t)*√t } * √[1 - (X₀(t)²/c²)]
2. Extraction:
   Crossing events occur when dX₀/dt changes sign.
   Record (tᵢ, X₀(tᵢ), θ(tᵢ)) for each crossing event.
3. Mapping:
   x_s = cos(θ)/max(|cos(θ)|,|sin(θ)|)
   y_s = sin(θ)/max(|cos(θ)|,|sin(θ)|)
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_astarte(T_total, dt, params):
    """
    Simulates Astarte dynamics over a total time T_total with time step dt.
    Returns arrays: time, x_A, x_B, X₀.
    
    Dynamics:
      θ(t) = ω*t + φ,   A = κ*λ*I,   Δ(t) = x_A(t) - x_B(t)
      x_A(t+dt) = X₀(t) + (1-μ)[x_A(t)-X₀(t)] + μ*A*cos(θ(t))
      x_B(t+dt) = X₀(t) + (1-μ)[x_B(t)-X₀(t)] + μ*A*sin(θ(t))
      X₀(t+dt) = X₀(t) + { ζ*(|I| - |X₀(t)|) - δ*Δ(t)*√t } * √[1 - (X₀(t)²/c²)]
    """
    num_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, num_steps)
    
    # Unpack parameters.
    I       = params['I']
    kappa   = params['kappa']
    lambda_ = params['lambda']
    mu      = params['mu']
    omega   = params['omega']
    phi     = params['phi']
    zeta    = params['zeta']
    delta   = params['delta']
    c       = params['c']
    
    A = kappa * lambda_ * I  # Scaled input
    
    # Initialize arrays.
    xA = np.zeros(num_steps)
    xB = np.zeros(num_steps)
    X0 = np.zeros(num_steps)
    
    # Set initial conditions.
    xA[0] = 0.0
    xB[0] = 0.0
    X0[0] = 0.0
    
    for t in range(num_steps - 1):
        current_time = time[t]
        theta = omega * current_time + phi
        
        # Update base registers.
        xA[t+1] = X0[t] + (1 - mu) * (xA[t] - X0[t]) + mu * A * np.cos(theta)
        xB[t+1] = X0[t] + (1 - mu) * (xB[t] - X0[t]) + mu * A * np.sin(theta)
        
        Delta = xA[t] - xB[t]
        # Compute relativistic damping factor.
        damp_factor = np.sqrt(max(0, 1 - (X0[t]**2 / c**2)))
        
        # Update null channel X₀.
        X0[t+1] = X0[t] + (zeta * (abs(I) - abs(X0[t])) - delta * Delta * np.sqrt(current_time + dt)) * damp_factor
    
    return time, xA, xB, X0

def extract_crossing_vector(time, X0, dt):
    """
    Computes the finite-difference derivative dX₀/dt and extracts indices where
    the sign of dX₀/dt changes (crossing events). Returns:
      - cross_times: times of the crossings,
      - cross_X0: corresponding X₀ values,
      - cross_indices: indices in the time array.
    """
    X0_velocity = np.diff(X0) / dt
    sign_changes = np.where(np.diff(np.sign(X0_velocity)) != 0)[0]
    cross_indices = sign_changes + 1  # Adjust index after diff.
    cross_times = time[cross_indices]
    cross_X0 = X0[cross_indices]
    return cross_times, cross_X0, cross_indices

def map_to_square(theta):
    """
    Maps a phase angle theta (radians) from the unit circle to square coordinates.
    Transformation:
      x_s = cos(theta) / max(|cos(theta)|, |sin(theta)|)
      y_s = sin(theta) / max(|cos(theta)|, |sin(theta)|)
    """
    x_c = np.cos(theta)
    y_c = np.sin(theta)
    denom = np.maximum(np.abs(x_c), np.abs(y_c))
    x_s = x_c / denom
    y_s = y_c / denom
    return x_s, y_s

def main():
    # --- Simulation Parameters ---
    params = {
        'I': 1.0,             # Constant input
        'kappa': 1.0,         # Scaling constant
        'lambda': 1.0,        # Scaling constant
        'mu': 0.1,            # Blend factor
        'omega': 2 * np.pi,   # Base angular frequency (radians per time unit)
        'phi': 0.0,           # Phase offset
        'zeta': 0.05,         # Feedforward adjustment factor
        'delta': 0.01,        # Capacitive correction strength
        'c': 1.0              # Energy-transfer cap
    }
    
    # --- Simulation Time ---
    # t = 0 is mapped to the calendar year 2025.
    T_total = 200    # Simulation time units (e.g., years)
    dt = 0.01
    
    # Run the Astarte simulation.
    time, xA, xB, X0 = simulate_astarte(T_total, dt, params)
    
    # Map simulation time to calendar years.
    years = 2025 + time
    
    # --- Extract the "Number Vector" from Crossing Events ---
    cross_times, cross_X0, cross_indices = extract_crossing_vector(time, X0, dt)
    cross_years = 2025 + cross_times  # Convert simulation time to years.
    
    print("Extracted 'Number Vector' from Crossing Events:")
    print("Index\tYear\t\tX₀ Value")
    for idx, yr, val in zip(cross_indices, cross_years, cross_X0):
        print(f"{idx:4d}\t{yr:8.2f}\t{val:12.6f}")
    
    # --- Map the Crossing Events to Square Coordinates ---
    omega = params['omega']
    phi = params['phi']
    cross_theta = omega * cross_times + phi
    x_square, y_square = map_to_square(cross_theta)
    
    print("\nMapped Square Coordinates for Crossing Events:")
    print("Index\tYear\t\tTheta (rad)\tX_square\tY_square")
    for idx, yr, theta_val, xs, ys in zip(cross_indices, cross_years, cross_theta, x_square, y_square):
        print(f"{idx:4d}\t{yr:8.2f}\t{theta_val:10.6f}\t{xs:10.6f}\t{ys:10.6f}")
    
    # --- Plotting ---
    # 1. Evolution of X₀ over time with crossing events marked.
    plt.figure(figsize=(10, 6))
    plt.plot(years, X0, label=r'$X_0(t)$')
    plt.xlabel("Year")
    plt.ylabel(r"$X_0(t)$")
    plt.title("Evolution of the Null Channel $X_0$ over Time")
    plt.scatter(cross_years, cross_X0, color='red', zorder=5, label="Crossing Events")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 2. Crossing events on the unit circle and their mapping to a square.
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Unit Circle Plot.
    theta_full = np.linspace(0, 2*np.pi, 400)
    ax[0].plot(np.cos(theta_full), np.sin(theta_full), 'k-', label="Unit Circle")
    ax[0].scatter(np.cos(cross_theta), np.sin(cross_theta), color='red', label="Crossing Points")
    ax[0].set_aspect('equal')
    ax[0].set_title("Crossing Points on the Circle")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend()
    ax[0].grid(True)
    
    # Square Mapping Plot.
    square_x = [-1, 1, 1, -1, -1]
    square_y = [-1, -1, 1, 1, -1]
    ax[1].plot(square_x, square_y, 'k-', label="Square Boundary")
    ax[1].scatter(x_square, y_square, color='blue', label="Mapped Crossing Points")
    ax[1].set_aspect('equal')
    ax[1].set_title("Mapped Crossing Points on the Square")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
```

---

## 4. Conclusion

This whitepaper presents a unified framework that ties the Astarte dynamics—characterized by helical, relativistic, energy‐transfer constraints—to square geometry via the extraction of critical crossing events. The mathematical formulation and simulation detailed herein demonstrate how circular dynamics can be mapped exactly to a square through the transformation of phase data at key transition points. This integrated approach not only reveals new invariants within dynamic systems but also provides a precise tool for measurement “from the inside or the outside” of a square over time.

We invite further exploration and validation of this framework across various dynamic systems and applications.

---

*End of Whitepaper*

*Thank you for your interest in Astarte. I hope this work sparks further inquiry and constructive dialogue. :Jazzhands:*

