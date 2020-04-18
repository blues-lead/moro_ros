#!/usr/bin/env python
import numpy as np
from quadprog import solve_qp
from math import sqrt


def smooth_path(path, num_steps=50, weight=1e-5, max_vel=0.5, max_acc=1.):
    """Return smooth position, velocity, acceleration, and jerk profiles, along
    with the associated time, for a path consisting of x,y coordinates defining
    line segments.

    Args:
        path (ndarray): Array of coordinates, size (n,2)
        num_steps (int, optional): Number of discretization steps for each line
            segment. Default: 50.
        weight (float, optional): Weight for adjusting tracking vs. smoothing.
            Default: 1e-5.
        max_vel (float, optional): Maximum velocity in m/s. Default: 0.5.
        max_acc (float, optional): Maximum acceleration in m/s**2. Default: 1.

    Returns:
        tuple: Position, velocity, acceleration, jerk, time. Time is size
            (n*num_steps,), other are size (n*num_steps,2)
    """
    # Form the variable vectors for a 4th order polynomial as a function of s,
    P = lambda s: np.array([[ 1,    s,      s**2,   s**3,   s**4 ]])  # position
    V = lambda s: np.array([[ 0,    1,      2*s,    3*s**3, 4*s**3 ]])  # velocity
    A = lambda s: np.array([[ 0,    0,      2,      6*s,    12*s**2  ]])  # acceleration
    J = lambda s: np.array([[ 0,    0,      0,      6,      12*s  ]])  # jerk

    # ... where s in the range [0,1]
    s = np.linspace(0, 1, num_steps)

    # The path consists of line segments connecting the individual coordinates
    num_segments = path.shape[0] - 1

    # The desired position on a line segment for any s is given by the line
    # equation
    pos_desired = lambda seg, s: (1-s)*seg+s*seg

    # The parameters to optimize are the coefficients of the polynomials for
    # the different line segments
    poly_order = 4
    num_params = (poly_order + 1)*num_segments
    param_idx = lambda seg: range(seg*(poly_order + 1),
                                  (seg + 1)*(poly_order + 1))

    # Formulate the problem in a way that it can be solved using quadratic
    # programming
    G_smooth = np.zeros((num_params, num_params))
    G_track = np.zeros((num_params, num_params))
    a = np.zeros((num_params, 2))

    for seg in range(num_segments):
        idx = param_idx(seg)
        for step in range(num_steps):
            G_smooth[np.ix_(idx, idx)] += np.dot(J(s[step]).T, J(s[step]))
            G_track[np.ix_(idx, idx)] += np.dot(P(s[step]).T, P(s[step]))
            a[idx, :] -= 2*P(s[step]).T*pos_desired(seg, s[step])
    G = 2*(weight*G_smooth + G_track)

    # Set equality constraints for the optimization
    # There are 4 global constraints and 3 for every line segment
    num_constraints = 4 + 3*(num_segments - 1)

    C = np.zeros((num_constraints, num_params))
    b = np.zeros((num_constraints, 2))

    # no clue how the solve_qp expects these values
    # but i gues it expects constraint to be equal to zero
    # TODO: set global constraints (start and end nodes)
    # Position at start must be same as path start
    C[0,0] = 1; C[0,1] = 0; C[0,2] = 0; C[0,3] = 0; C[0,4] = 0
    b[0] = path[0]
    # Position at end must be the same as last path node position
    C[1,-5] = 1; C[1,-4] = 1; C[1,-3] = 1; C[1,-2] = 1; C[1,-1] = 1
    b[1] = path[-1]
    # Velocity at start must be zero
    C[2,1] = 1; C[2,2] = 0; C[2,3] = 0; C[2,4] = 0
    b[2] = 0
    # Velocity at end must be zero
    C[3,-4] = 1; C[3,-3] = 2; C[3,-2] = 3; C[3,-1] = 4
    b[3] = 0

    # TODO: Assert smooth transition between line segments
    for i in range(0,num_segments-1):
        # For end of each node Pos, vel, and acc must be the same as the start of next node

        # Pos
        C[i*3+4, 5*i] = 1; C[i*3+4, 5*i+1] = 1; C[i*3+4, 5*i+2] = 1; C[i*3+4, 5*i+3] = 1; C[i*3+4, 5*i+4] = 1
        C[i*3+4, 5*i+5] = -1; C[i*3+4, 5*i+6] = 0; C[i*3+4, 5*i+7] = 0; C[i*3+4, 5*i+8] = 0; C[i*3+4, 5*i+9] = 0

        # Vel
        C[i*3+4+1, 5*i+1] = 1; C[i*3+4+1, 5*i+2] = 2; C[i*3+4+1, 5*i+3] = 3; C[i*3+4+1, 5*i+4] = 4
        C[i*3+4+1, 5*i+6] = -1; C[i*3+4+1, 5*i+7] = 0; C[i*3+4+1, 5*i+8] = 0; C[i*3+4+1, 5*i+9] = 0

        # Acc
        C[i*3+4+2, 5*i+2] = 2; C[i*3+4+2, 5*i+3] = 6; C[i*3+4+2, 5*i+4] = 12
        C[i*3+4+2, 5*i+7] = -2; C[i*3+4+2, 5*i+8] = 0; C[i*3+4+2, 5*i+9] = 0

    # Solve the polynomial coefficients using quadratic programming
    poly_coef = np.zeros((num_params, 2))
    poly_coef[:, 0] = solve_qp(G, -a[:, 0], -C.T, -b[:, 0],
                               meq=num_constraints)[0]
    poly_coef[:, 1] = solve_qp(G, -a[:, 1], -C.T, -b[:, 1],
                               meq=num_constraints)[0]

    # Calculate the smooth trajectory using the coefficients
    n = num_segments*num_steps

    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    acc = np.zeros((n, 2))
    jerk = np.zeros((n, 2))
    time = np.zeros(n)

    for seg in range(num_segments):
        idx = param_idx(seg)
        for step in range(num_steps):
            i = seg*num_steps + step
            time[i] = seg + s[step]
            pos[i, :] = np.dot(P(s[step]), poly_coef[idx, :])
            vel[i, :] = np.dot(V(s[step]), poly_coef[idx, :])
            acc[i, :] = np.dot(A(s[step]), poly_coef[idx, :])
            jerk[i, :] = np.dot(J(s[step]), poly_coef[idx, :])

    # Remove duplicate entries at beginning of segments
    valid_idx = np.unique(time, return_index=True)[1]
    pos = pos[valid_idx, :]
    vel = vel[valid_idx, :]
    acc = acc[valid_idx, :]
    jerk = jerk[valid_idx, :]
    time = time[valid_idx]

    # Time scaling for constraint on maximum velocity
    # Scale also upwards to utilize the full capacity of the base
    abs_vel= []
    for v in vel:
        abs_vel.append(np.linalg.norm([v[0], v[1]]))

    scaling_factor = max_vel/max(abs_vel)

    vel = scaling_factor*vel
    acc = scaling_factor**2*acc
    jerk = scaling_factor**3*jerk
    time = time/scaling_factor

    # Time scaling for constraint on maximum acceleration
    # Only scale down if limits are exceeded

    # TODO
    abs_acc= []
    for a in acc:
        abs_acc.append(np.linalg.norm([a[0], a[1]]))

    scaling_factor = max_acc/max(abs_acc)
    # only scale down
    if scaling_factor < 1:
        vel = sqrt(scaling_factor)*vel
        acc = scaling_factor*acc
        jerk = sqrt(scaling_factor)*scaling_factor*jerk
        time = time/sqrt(scaling_factor)


    return pos, vel, acc, jerk, time

if __name__ == "__main__":
    p = np.array([[4.26271167, 1.90129996],
        [6.97728825, 2.16089496],
        [5.94148132, 0.3774518 ],
        [4.35734741, 0.57725099],
        [2.79453604, 1.22789253],
        [3.28322137, 3.22574976],
        [4.28552018, 2.78609033],
        [7.33380168, 4.0844386 ],
        [9.07464646, 1.85209234],
        [9.4966894, 7.2826944 ],
        [9.35290857, 9.19987348],
        [5.27908823, 9.37571584],
        [7.32593449, 8.94651292]])

    print(smooth_path(p))
