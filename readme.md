This project should allow the user to build simple microstructures in 2D and 3D and investigate different load cases.
The underlying framework is GooseFEM by Tom de Geus and can be found on GitHub (https://github.com/tdegeus).
For an educational purpose, simple material models are implemented in this project too. Up to now, there are the following models:
A hyperelastic material model for finite strains taking the deformation gradient w.r.t. the reference configuration
and calculates the stress and tangent matrix w.r.t. to reference configuration. It also calculates the Cauchy stress for plotting purposes.
A elastoplasstic material model is under construction.

Several test cases with simple microstructures are provided as educational examples.


*/ Use (GitHubFlavoredMarkdown) for this