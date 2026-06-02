# PINN advantage over traditional is that data is not required to be in a mesh. can run some comparison tests based on this idea. with full data classical methods are likely faster.

# generally data comes from boundary or initial conditions, and then collocation points are sampled from domain

# Implement L2RE Done

# Study problems were boundary conditions are non uniform. eg laplace with sin on top edge. seems like the base line model preforms significantly worse. Retry after fixing model definition mistake

# Investigate impact of domain 

# Do some loss landscape visualization

# Try some spectral bias experiments

# Try some NTK experiments 

# Option dynamic inverse problem

# test how number data points affect data loss noise