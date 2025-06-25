=====
Units
=====

*Aperture* uses a unit system based on the `Heaviside-Lorentz units <https://en.wikipedia.org/wiki/Heaviside%E2%80%93Lorentz_units>`_. In the HL units, Maxwell equations look like:

.. math::
   \nabla\cdot\mathbf{E} &= \rho \\
   \nabla\cdot\mathbf{B} &= 0 \\
   \frac{\partial \mathbf{E}}{\partial t} &= c\nabla\times\mathbf{B} - \mathbf{J} \\
   \frac{\partial \mathbf{B}}{\partial t} &= -c\nabla\times\mathbf{E}

In addition, we need to choose a length scale and time scale to make the
equations dimensionless. We demand that :math:`c = 1`, therefore choosing a
length scale automatically determines the appropriate time scale, and vice
versa. Energy is measured using units of :math:`m_ec^2`, while particle momentum
is measured in units of :math:`m_ec`. If we specify a unit of time as :math:`t_0
= \omega_0^{-1}`, the Lorentz force then takes the dimensionless form:

.. math::
   \frac{d\mathbf{p}}{m_ec\omega_0 dt} = \frac{e}{m_ec\omega_0}\left(\mathbf{E} + \frac{\mathbf{v}}{c}\times\mathbf{B}\right),

It is therefore appropriate to use :math:`eE/m_ec\omega_0` as the dimensionless
form of the electric (or magnetic) field. The dimensionless value of the
magnetic field is therefore equal to the dimensionless ratio
:math:`\omega_B/\omega_0`, or the dimensionless electron cyclotron frequency.
The dimensionless charge density can be defined using the Gauss's law
:math:`\nabla\cdot\mathbf{E}=\rho`:

.. math::
   \frac{eE}{m_ec\omega_0}\frac{c/\omega_0}{x} = \frac{e\rho}{m_e\omega_0^2}.

The right hand side is therefore the natural definition of dimensionless charge
density. In the code units, the dimensionless plasma frequency is nicely related
to the charge density as follows:

.. math::
   \omega_p \equiv \sqrt{\frac{ne^2}{m_e}} = \sqrt{\frac{e\rho}{m_e\omega_0^2}}\omega_0.

In other words, the dimensionless plasma frequency is simply the square root of
the dimensionless charge density, :math:`\tilde{\omega}_p =
\sqrt{\tilde{\rho}}`. Tilde here means the dimensionless quantity. This relationship is very convenient to use in plasma
simulations.

In HL units, the energy density of electric/magnetic field is
:math:`\mathbf{E}^2/2` or :math:`\mathbf{B}^2/2` respectively, in contrast with
the corresponding expressions of :math:`\mathbf{E}^2/8\pi` and
:math:`\mathbf{B}^2/8\pi` in CGS units. The magnetization :math:`\sigma` is now defined as:

.. math::
   \sigma \equiv \frac{B^2}{nm_ec^2} = \frac{(eB/m_ec\omega_0)^2}{e^2nm_ec^2}m_e^2c^2\omega_0^2 = \frac{(eB/m_ec\omega_0)^2}{e\rho/m_e\omega_0^2}.

Identifying the terms, one can see that :math:`\sigma = \tilde{B}^2/\tilde{\rho}`.

In an actual simulation, one needs to specify either a length scale :math:`x_0` or a time scale :math:`\omega_0^{-1}`, and it will completely specify the dimensionless unit system. For example, for a global pulsar simulation, the natural length scale is either the stellar radius :math:`R_*`, or the light cylinder radius :math:`R_\mathrm{LC}`. For a black hole simulation the natural length scale may be the gravitational radius :math:`r_g`. For a reconnection simulation it is often okay to choose the upstream :math:`\omega_0 = \omega_p`. In that case, one simply needs to initialize the upstream plasma density such that :math:`\tilde{\rho} = 1`.
