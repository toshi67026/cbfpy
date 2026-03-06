Theory
======

This page provides the mathematical background for Control Barrier Functions (CBFs)
implemented in cbfpy.


Control Barrier Functions
-------------------------

A Control Barrier Function (CBF) provides formal safety guarantees for dynamical systems.
Consider an affine control system:

.. math::

   \dot{x} = f(x) + g(x) u

where :math:`x \in \mathbb{R}^n` is the state and :math:`u \in \mathbb{R}^m` is the control input.

A **safe set** :math:`\mathcal{C}` is defined as the 0-superlevel set of a continuously
differentiable function :math:`h : \mathbb{R}^n \to \mathbb{R}`:

.. math::

   \mathcal{C} = \{ x \in \mathbb{R}^n \mid h(x) \geq 0 \}

The function :math:`h` is a CBF if there exists an extended class-:math:`\mathcal{K}` function
:math:`\alpha` such that for all :math:`x \in \mathcal{C}`:

.. math::

   \sup_{u} \left[ \frac{\partial h}{\partial x} (f(x) + g(x) u) + \alpha(h(x)) \right] \geq 0


CBF-QP Formulation
------------------

To find the safe input closest to a desired nominal input :math:`u_{\text{nom}}`,
we solve a Quadratic Program (QP):

.. math::

   \min_{u} \quad & \frac{1}{2} (u - u_{\text{nom}})^T P (u - u_{\text{nom}}) \\
   \text{s.t.} \quad & \frac{\partial h}{\partial x} g(x) \, u + \alpha(h(x)) \geq 0

In cbfpy, we denote:

- :math:`G = \frac{\partial h}{\partial x} g(x)` — the constraint gradient (computed by each CBF class)
- :math:`\alpha(h)` — the class-K function output (default: :math:`\alpha(h) = h`)
- :math:`P` — the weight matrix (positive definite)

Multiple CBFs can be combined by stacking their constraints.


Barrier Functions in cbfpy
--------------------------

Scalar CBF
^^^^^^^^^^

For a 1D state :math:`x` with limit :math:`l`:

.. math::

   h(x) = \text{sign} \cdot (x - l)

where :math:`\text{sign} = +1` for ``keep_upper=True`` (stay above :math:`l`)
and :math:`\text{sign} = -1` for ``keep_upper=False`` (stay below :math:`l`).

Scalar Range CBF
^^^^^^^^^^^^^^^^

For a 1D state :math:`x` within range :math:`[a, b]`:

.. math::

   h(x) = \text{sign} \cdot \left[ \left( \frac{b - a}{2} \right)^2 - \left( x - \frac{a + b}{2} \right)^2 \right]

This is positive inside the range and negative outside.

Circle CBF
^^^^^^^^^^

For a 2D agent position :math:`p` with a circular region of center :math:`c` and radius :math:`r`:

.. math::

   h(p) = \text{sign} \cdot \left( 1 - \left\| \frac{p - c}{r} \right\|_2 \right)

where :math:`\text{sign} = +1` for ``keep_inside=True``.

P-norm 2D CBF
^^^^^^^^^^^^^^

Generalizes the circle to a p-norm shaped region. For center :math:`c`,
semi-axes :math:`w`, rotation :math:`\theta`, and norm order :math:`p`:

.. math::

   \tilde{p} = R(-\theta) \frac{p - c}{w}

.. math::

   h(\tilde{p}) = \text{sign} \cdot \left( 1 - \| \tilde{p} \|_p \right)

where :math:`R(\theta)` is the 2D rotation matrix. Setting :math:`p = 2` gives an ellipse,
:math:`p = 1` gives a diamond, and :math:`p \to \infty` gives a rectangle.

Unicycle Variants
^^^^^^^^^^^^^^^^^

For unicycle models with state :math:`(x, y, \theta)` and input :math:`(v, \omega)`,
the constraint gradient is transformed by the input matrix:

.. math::

   A = \begin{pmatrix} \cos\theta & 0 \\ \sin\theta & 0 \\ 0 & 1 \end{pmatrix}

so that the CBF constraint becomes:

.. math::

   \frac{\partial h}{\partial x} A \, u + \alpha(h) \geq 0

LiDAR CBF
^^^^^^^^^^

For LiDAR-based obstacle avoidance, each laser ray at range :math:`r` and angle :math:`\phi`
defines a barrier function:

.. math::

   h(r, \phi) = r - r_c(\phi)

where :math:`r_c(\phi) = \sqrt{(w_x \cos\phi)^2 + (w_y \sin\phi)^2}` is the
critical distance defined by the ellipsoidal safety margin with semi-axes :math:`(w_x, w_y)`.


CBFController
-------------

The :class:`~cbfpy.cbf_controller.CBFController` class simplifies the common pattern of
combining multiple CBFs with a QP solver:

.. code-block:: python

   from cbfpy import CBFController, CircleCBF

   # Define CBFs
   cbf1 = CircleCBF(center1, radius1, keep_inside=False)
   cbf2 = CircleCBF(center2, radius2, keep_inside=False)

   # Create controller
   controller = CBFController([cbf1, cbf2], P=np.eye(2))

   # In control loop:
   cbf1.calc_constraints(agent_position)
   cbf2.calc_constraints(agent_position)
   status, safe_input = controller.optimize(nominal_input)
