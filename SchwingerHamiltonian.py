import numpy as np
from qiskit.opflow import Z, X, Y, I  # Pauli Z, X matrices and identity


class SchwingerHamiltonian:
    def __init__(self, number_of_sites=2, x=25., mu=0.1, l=[]):
        self.number_of_sites = number_of_sites
        self.x = x
        self.mu = mu
        if not l:
            self.l = np.zeros(self.number_of_sites)
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Id = I
        self.Sp = 1/2 * (self.X + 1j * self.Y)
        self.Sm = 1/2 * (self.X - 1j * self.Y)

    def fermion_kinetic_term(self):
        k = 0
        for n in range(self.number_of_sites - 1):
            k += self.id_rep(n) ^ self.Sp ^ self.Sm ^ self.id_rep(self.number_of_sites - 1 - n - 1)
            k += self.id_rep(n) ^ self.Sm ^ self.Sp ^ self.id_rep(self.number_of_sites - n - 2)
        return self.x * k

    def fermion_mass_term(self):
        m = 0
        for n in range(self.number_of_sites):
            m += (-1)**n * self.id_rep(n) ^ self.Z ^ self.id_rep(self.number_of_sites - n - 1)
            m += self.id_rep(self.number_of_sites)
        return self.mu / 2 * m

    def gauge_kinetic_term(self):
        g = 0
        for n in range(self.number_of_sites - 1):
            sub_g = 0
            for k in range(n + 1):
                sub_g += self.id_rep(k) ^ self.Z ^ self.id_rep(self.number_of_sites - k - 1)
                term = (-1 * self.id_rep(self.number_of_sites)) ** k if k > 0 else self.id_rep(self.number_of_sites)
                sub_g += 1/2 * term
            g += (self.id_rep(self.number_of_sites) + sub_g) ** 2
        return g

    def hamiltonian(self):
        h = self.fermion_kinetic_term() + self.fermion_mass_term() + self.gauge_kinetic_term()
        return h

    def id_rep(self, n):
        result = 1
        for i in range(n):
            result = result ^ self.Id
        return result
