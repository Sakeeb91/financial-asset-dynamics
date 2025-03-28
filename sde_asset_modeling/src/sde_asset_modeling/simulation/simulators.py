import numpy as np

def euler_maruyama(sde, x0, t_span, dt, random_state=None):
    """
    Simulate an SDE using the Euler-Maruyama method.
    
    dX_t = a(X_t, t) dt + b(X_t, t) dW_t
    
    The Euler-Maruyama discretization is:
    X_{t+dt} = X_t + a(X_t, t) * dt + b(X_t, t) * sqrt(dt) * Z
    where Z ~ N(0, 1)
    
    Args:
        sde: The SDE object with drift and diffusion methods
        x0 (float): Initial state
        t_span (tuple): (t_start, t_end) time interval
        dt (float): Time step size
        random_state (int, optional): Seed for random number generator
        
    Returns:
        tuple: (t, x) where t is an array of time points and x is an array of state values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    
    # Initialize arrays for time and state
    t = np.linspace(t_start, t_end, n_steps)
    x = np.zeros(n_steps)
    x[0] = x0
    
    # Euler-Maruyama iteration
    for i in range(1, n_steps):
        t_current = t[i-1]
        x_current = x[i-1]
        
        # Generate random normal increment
        dw = np.random.normal(0, np.sqrt(dt))
        
        # Euler-Maruyama update
        x[i] = x_current + sde.drift(x_current, t_current) * dt + \
               sde.diffusion(x_current, t_current) * dw
        
    return t, x

def milstein(sde, x0, t_span, dt, random_state=None):
    """
    Simulate an SDE using the Milstein method.
    
    dX_t = a(X_t, t) dt + b(X_t, t) dW_t
    
    The Milstein discretization is:
    X_{t+dt} = X_t + a(X_t, t) * dt + b(X_t, t) * sqrt(dt) * Z + 
                0.5 * b(X_t, t) * b'(X_t, t) * ((sqrt(dt) * Z)^2 - dt)
    where Z ~ N(0, 1) and b' is the derivative of b with respect to x
    
    Args:
        sde: The SDE object with drift, diffusion, and diffusion_derivative methods
        x0 (float): Initial state
        t_span (tuple): (t_start, t_end) time interval
        dt (float): Time step size
        random_state (int, optional): Seed for random number generator
        
    Returns:
        tuple: (t, x) where t is an array of time points and x is an array of state values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    
    # Initialize arrays for time and state
    t = np.linspace(t_start, t_end, n_steps)
    x = np.zeros(n_steps)
    x[0] = x0
    
    # Milstein iteration
    for i in range(1, n_steps):
        t_current = t[i-1]
        x_current = x[i-1]
        
        # Generate random normal increment
        dw = np.random.normal(0, np.sqrt(dt))
        
        # Milstein update
        diffusion_val = sde.diffusion(x_current, t_current)
        diffusion_deriv = sde.diffusion_derivative(x_current, t_current)
        
        x[i] = x_current + sde.drift(x_current, t_current) * dt + \
               diffusion_val * dw + \
               0.5 * diffusion_val * diffusion_deriv * (dw**2 - dt)
        
    return t, x

def generate_paths(simulator, sde, x0, t_span, dt, n_paths=1, random_state=None):
    """
    Generate multiple simulation paths using the specified simulator method.
    
    Args:
        simulator: Simulation function (euler_maruyama or milstein)
        sde: The SDE object with drift and diffusion methods
        x0 (float): Initial state
        t_span (tuple): (t_start, t_end) time interval
        dt (float): Time step size
        n_paths (int): Number of simulation paths to generate
        random_state (int, optional): Seed for random number generator
        
    Returns:
        tuple: (t, x) where t is an array of time points and 
              x is a 2D array with shape (n_paths, len(t))
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    
    # Initialize time array and paths matrix
    t = np.linspace(t_start, t_end, n_steps)
    x = np.zeros((n_paths, n_steps))
    
    # Generate n_paths simulation paths
    for i in range(n_paths):
        # Use different random seed for each path
        path_seed = None if random_state is None else random_state + i
        _, x[i, :] = simulator(sde, x0, t_span, dt, random_state=path_seed)
    
    return t, x 