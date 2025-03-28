import numpy as np
import matplotlib.pyplot as plt
from .simulators import euler_maruyama, milstein, generate_paths

class SimulationEngine:
    """
    Engine for running SDE simulations and comparing different methods.
    """
    
    def __init__(self, sde, x0, t_span, dt):
        """
        Initialize the simulation engine.
        
        Args:
            sde: SDE object with drift and diffusion methods
            x0 (float): Initial state
            t_span (tuple): (t_start, t_end) time interval
            dt (float): Time step size
        """
        self.sde = sde
        self.x0 = x0
        self.t_span = t_span
        self.dt = dt
        
    def run_simulation(self, method='euler', n_paths=1, random_state=None):
        """
        Run a simulation using the specified method.
        
        Args:
            method (str): Simulation method ('euler' or 'milstein')
            n_paths (int): Number of simulation paths
            random_state (int, optional): Seed for random number generator
            
        Returns:
            tuple: (t, x) where t is an array of time points and 
                  x is array of state values (1D for single path, 2D for multiple paths)
        """
        if method.lower() == 'euler':
            simulator = euler_maruyama
        elif method.lower() == 'milstein':
            simulator = milstein
        else:
            raise ValueError(f"Unknown simulation method: {method}")
        
        if n_paths == 1:
            t, x = simulator(self.sde, self.x0, self.t_span, self.dt, random_state)
        else:
            t, x = generate_paths(simulator, self.sde, self.x0, self.t_span, self.dt, 
                                  n_paths, random_state)
        
        return t, x
    
    def run_exact_solution(self, n_paths=1, random_state=None):
        """
        Generate paths using the exact analytical solution if available.
        
        Args:
            n_paths (int): Number of paths to generate
            random_state (int, optional): Seed for random number generator
            
        Returns:
            tuple: (t, x) where t is an array of time points and 
                  x is array of state values (1D for single path, 2D for multiple paths)
        """
        if not hasattr(self.sde, 'exact_solution'):
            raise NotImplementedError("Exact solution not implemented for this SDE")
        
        t_start, t_end = self.t_span
        n_steps = int((t_end - t_start) / self.dt) + 1
        t = np.linspace(t_start, t_end, n_steps)
        
        if random_state is not None:
            np.random.seed(random_state)
        
        if n_paths == 1:
            x = self.sde.exact_solution(self.x0, t, random_state)
        else:
            x = np.zeros((n_paths, len(t)))
            for i in range(n_paths):
                path_seed = None if random_state is None else random_state + i
                x[i, :] = self.sde.exact_solution(self.x0, t, path_seed)
        
        return t, x
    
    def compare_methods(self, n_paths=1, random_state=None, exact_solution=True):
        """
        Compare different simulation methods with each other and optionally with the exact solution.
        
        Args:
            n_paths (int): Number of paths to simulate
            random_state (int, optional): Seed for random number generator
            exact_solution (bool): Whether to include the exact solution
            
        Returns:
            dict: Dictionary containing simulation results for each method
        """
        results = {}
        
        # Run Euler-Maruyama simulation
        results['euler'] = self.run_simulation('euler', n_paths, random_state)
        
        # Run Milstein simulation
        results['milstein'] = self.run_simulation('milstein', n_paths, random_state)
        
        # Include exact solution if requested and available
        if exact_solution and hasattr(self.sde, 'exact_solution'):
            results['exact'] = self.run_exact_solution(n_paths, random_state)
        
        return results
    
    def plot_paths(self, results=None, methods=None, n_paths=1, random_state=None, 
                   figsize=(10, 6), title=None, ylabel=None):
        """
        Plot simulation paths for different methods.
        
        Args:
            results (dict, optional): Pre-computed simulation results
            methods (list, optional): Methods to plot
            n_paths (int): Number of paths to simulate if results not provided
            random_state (int, optional): Seed for random number generator
            figsize (tuple): Figure size
            title (str, optional): Plot title
            ylabel (str, optional): Y-axis label
            
        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        if results is None:
            has_exact = hasattr(self.sde, 'exact_solution')
            results = self.compare_methods(n_paths, random_state, has_exact)
        
        if methods is None:
            methods = list(results.keys())
        else:
            methods = [m for m in methods if m in results]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for method in methods:
            t, x = results[method]
            
            if n_paths == 1:
                ax.plot(t, x, label=method.capitalize())
            else:
                for i in range(min(n_paths, 5)):  # Plot at most 5 paths to avoid clutter
                    ax.plot(t, x[i], alpha=0.5, 
                            label=f"{method.capitalize()} (path {i+1})" if i == 0 else None)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel if ylabel is not None else 'State')
        ax.set_title(title if title is not None else 'SDE Simulation Paths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def compare_error(self, dt_values, final_time_only=True, n_paths=100, random_state=None):
        """
        Compare simulation errors for different time step sizes.
        
        Args:
            dt_values (list): List of time step sizes to compare
            final_time_only (bool): Whether to compute error at final time only or over all time points
            n_paths (int): Number of paths to simulate for error estimation
            random_state (int, optional): Seed for random number generator
            
        Returns:
            dict: Dictionary containing error statistics for each method and dt
        """
        if not hasattr(self.sde, 'exact_solution'):
            raise NotImplementedError("Exact solution not implemented for this SDE")
        
        t_start, t_end = self.t_span
        errors = {'euler': [], 'milstein': []}
        
        for dt in dt_values:
            # Temporary simulation engine with current dt
            temp_engine = SimulationEngine(self.sde, self.x0, (t_start, t_end), dt)
            
            # Generate exact solution as reference
            t_exact, x_exact = temp_engine.run_exact_solution(n_paths, random_state)
            
            # Run simulations
            for method in ['euler', 'milstein']:
                t_sim, x_sim = temp_engine.run_simulation(method, n_paths, random_state)
                
                # Compute error
                if final_time_only:
                    # Error at final time only
                    err = np.abs(x_sim[:, -1] - x_exact[:, -1])
                else:
                    # Root mean squared error over all time points
                    err = np.sqrt(np.mean((x_sim - x_exact)**2, axis=1))
                
                # Compute error statistics
                errors[method].append({
                    'dt': dt,
                    'mean': np.mean(err),
                    'std': np.std(err),
                    'median': np.median(err),
                    'max': np.max(err),
                    'min': np.min(err)
                })
        
        return errors 