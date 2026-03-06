Examples
========

All examples can be run with:

.. code-block:: bash

   python examples/example_{name}.py

Each example creates a CBF optimizer wrapping a CBF and QP solver,
then runs a matplotlib animation showing the nominal input (black arrows)
vs. the CBF-filtered safe input (red arrows).


Scalar CBF
----------

Constrains a 1D state variable to stay above or below a limit.
The limit and direction change over time to demonstrate dynamic reconfiguration.

.. literalinclude:: ../../examples/example_scalar_cbf.py
   :language: python
   :caption: examples/example_scalar_cbf.py


Scalar Range CBF
----------------

Constrains a 1D state variable to stay inside or outside a range ``[a, b]``.
Uses the barrier function :math:`h(x) = \left(\frac{b-a}{2}\right)^2 - \left(x - \frac{a+b}{2}\right)^2`.

.. literalinclude:: ../../examples/example_scalar_range_cbf.py
   :language: python
   :caption: examples/example_scalar_range_cbf.py


Circle CBF
----------

Two agents move toward each other with a circular collision avoidance constraint.
Each agent treats the other's position as an obstacle with ``keep_inside=False``.

.. literalinclude:: ../../examples/example_circle_cbf.py
   :language: python
   :caption: examples/example_circle_cbf.py


P-norm 2D CBF
-------------

An agent avoids a p-norm (ellipsoidal) shaped area.
The p-norm shape generalizes circles (:math:`p=2`), diamonds (:math:`p=1`), and rectangles (:math:`p \to \infty`).

.. literalinclude:: ../../examples/example_pnorm2d_cbf.py
   :language: python
   :caption: examples/example_pnorm2d_cbf.py


Unicycle Circle CBF
-------------------

Circular area constraint for a unicycle model (input: linear velocity + angular velocity).
The unicycle kinematics are handled by an input transformation matrix.

.. literalinclude:: ../../examples/example_unicycle_circle_cbf.py
   :language: python
   :caption: examples/example_unicycle_circle_cbf.py


Unicycle P-norm 2D CBF
----------------------

P-norm shaped area constraint for a unicycle model.
Combines the p-norm barrier function with unicycle input transformation.

.. literalinclude:: ../../examples/example_unicycle_pnorm2d_cbf.py
   :language: python
   :caption: examples/example_unicycle_pnorm2d_cbf.py


LiDAR CBF
----------

LiDAR-based obstacle avoidance for a unicycle model.
Simulates a 2D LiDAR sensor and creates a CBF constraint for each ray,
enabling reactive navigation through environments with multiple obstacles.

.. literalinclude:: ../../examples/example_lidar_cbf.py
   :language: python
   :caption: examples/example_lidar_cbf.py


Multi-Agent Collision Avoidance
-------------------------------

Four agents swap positions diagonally while avoiding each other.
Uses :class:`~cbfpy.cbf_controller.CBFController` with multiple ``CircleCBF`` instances per agent.

.. literalinclude:: ../../examples/example_multi_agent_cbf.py
   :language: python
   :caption: examples/example_multi_agent_cbf.py


Path Following with Obstacle Avoidance
---------------------------------------

An agent follows a sequence of waypoints while avoiding circular obstacles.
Demonstrates :class:`~cbfpy.cbf_controller.CBFController` composing multiple CBF constraints.

.. literalinclude:: ../../examples/example_path_following_cbf.py
   :language: python
   :caption: examples/example_path_following_cbf.py
