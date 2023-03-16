from SchwingerHamiltonian import *

from qiskit.providers.aer import QasmSimulator
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2

from numpy import linalg
import matplotlib.pyplot as plt

# Instantiate the SchwingerHamiltonian
number_of_sites = 4
Hamiltonian = SchwingerHamiltonian(number_of_sites=number_of_sites)

# Get the Hamiltonian
h = Hamiltonian.hamiltonian()

# you can swap this for a real quantum device and keep the rest of the code the same!
backend = QasmSimulator()

# COBYLA usually works well for small problems like this one
optimizer = COBYLA(maxiter=200)

# EfficientSU2 is a standard heuristic chemistry ansatz from Qiskit's circuit library
ansatz = EfficientSU2(number_of_sites, reps=3)

# Callback to get partial cost
counts = []
values = []


def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)


# set the algorithm
vqe = VQE(ansatz, optimizer, callback=store_intermediate_result, quantum_instance=backend)

# run it with the Hamiltonian we defined above
result = vqe.compute_minimum_eigenvalue(h)

# print the result (it contains lots of information)
print(result)

plt.plot(counts, values, label='COBYLA')
plt.xlabel('Eval count')
plt.ylabel('Energy')
plt.title('Energy convergence')
print(np.shape(h.to_matrix()))
eigen_values, eigen_vectors = linalg.eig(h.to_matrix())
plt.axhline(y=min(eigen_values), color='r', linestyle='-', label='analytical')
plt.legend(loc='upper right')

plt.show()
