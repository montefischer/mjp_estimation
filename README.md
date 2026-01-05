# Parametric Estimation for Time-Inhomogeneous Markov Jump Processes

This library implements parametric estimation techniques for time-inhomogeneous parametric Markov jump process models, with an emphasis on networks of queues. For an example of the library in action, see the `notebooks` folder. 

The core functionality of the library is contained in the following files under `src/mjp_estimation`.
* `base_functionality.py` - Specifies core classes for the library, including the `MJPModel` base class, the `Simulator` class and classes associated with sample paths, and the `JointKolmogorovSolver` class for ODE solving to obtain transient probabilities and expectations associated with a Markov jump process model.
* `estimators.py` - Parametric estimators, split between estimators implementing deterministic ODE solving methods, and estimators implementing stochastic gradient descent with likelihood ratio gradients.
* `objectives.py` - Specification of the objective functions used by the parametric estimation classes, including likelihood-based objectives and conditional least squares objectives.
* `optimizers.py` - Implementations of the optimization methods used by the parametric estimation classes.
* `jackson_model.py` - Implementations of a variety of parameterized networks of queues with various specifications of arrival rates and service rates. Model constructors are implemented so that models can be initialized quickly even for large state space.
* `plots.py` - Helper functions for plotting.

This library is in beta. Methods and classes are subject to change.