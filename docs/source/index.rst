cbfpy documentation
===================

**cbfpy** is a Python package for using simple Control Barrier Functions (CBF).

CBF provides a framework for safety-constrained control by solving the optimization problem:

.. math::

   \min_{u} \quad & \frac{1}{2} (u - u_{\text{nom}})^T P (u - u_{\text{nom}}) \\
   \text{s.t.} \quad & \frac{\partial h}{\partial x} u + \alpha(h(x)) \geq 0

where :math:`h(x)` is the barrier function defining the safe set :math:`\{x \mid h(x) \geq 0\}`,
and :math:`\alpha` is an extended class-K function.

Features
--------

- Symbolic CBF construction using SymPy (automatic gradient computation)
- Multiple CBF types: scalar, circular, p-norm, unicycle, LiDAR-based
- Lightweight QP solver via `quadprog <https://pypi.org/project/quadprog/>`_ (Goldfarb/Idnani algorithm)
- Runtime parameter reconfiguration via ``set_parameters()``
- :class:`~cbfpy.cbf_controller.CBFController` for easy multi-CBF composition

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from cbfpy import CircleCBF, CBFNomQPSolver

   # Create a circular CBF (keep agent inside a circle)
   cbf = CircleCBF(center=np.zeros(2), radius=2.0, keep_inside=True)

   # Compute constraints at the current agent position
   cbf.calc_constraints(agent_position=np.array([1.5, 0.0]))
   G, alpha_h = cbf.get_constraints()

   # Solve the CBF-QP to get the safe input
   solver = CBFNomQPSolver()
   nominal_input = np.array([1.0, 0.0])  # desired direction
   status, safe_input = solver.optimize(nominal_input, np.eye(2), [G], [alpha_h])

Or using :class:`~cbfpy.cbf_controller.CBFController` for a simpler API:

.. code-block:: python

   from cbfpy import CBFController, CircleCBF

   cbf = CircleCBF(center=np.zeros(2), radius=2.0, keep_inside=True)
   controller = CBFController([cbf], P=np.eye(2))

   cbf.calc_constraints(agent_position=np.array([1.5, 0.0]))
   status, safe_input = controller.optimize(nominal_input=np.array([1.0, 0.0]))


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   theory
   cbfpy
   examples
