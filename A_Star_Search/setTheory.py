# Set Operations
A = {2, 36, 4, 6, 7, 8, 11, 13, 10, 3}
B = {1, 3, 7, 13, 21, 31, 22}
C = {13, 22, 7, 3, 11, 6, 1, 14, 19, 5}
D = {23, 14, 8, 7, 5}

# a. A ∪ B
union_A_B = A.union(B)
print("A ∪ B =", union_A_B)

# b. B ∩ C
intersection_B_C = B.intersection(C)
print("B ∩ C =", intersection_B_C)

# c. C - D
difference_C_D = C.difference(D)
print("C - D =", difference_C_D)

# d. C x D (Cartesian product)
cartesian_C_D = {(c, d) for c in C for d in D}
print("C × D =", cartesian_C_D)

# Relations:
# R1
R1 = {('a', 'a'), ('a', 'b'), ('c', 'b'), ('c', 'e'), ('d', 'b'), ('d', 'e'), ('e', 'e')}

# R2
R2 = {('q1', 'q2'), ('q2', 'q3'), ('q2', 'q4'), ('q3', 'q6'), ('q4', 'q5'), ('q4', 'q3'),
('q5', 'q4'), ('q6', 'q6'), ('q6', 'q5')}

# Function to plot the relation graph
import networkx as nx
import matplotlib.pyplot as plt

def plot_relation(relation, title):
    G = nx.DiGraph()
    for start, end in relation:
        G.add_edge(start, end)
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=2000, node_color='lightblue', arrowsize=20)
    plt.title(title)
    plt.show()

# Plot R1
plot_relation(R1, "Graph Representation of Relation R1")

# Plot R2
plot_relation(R2, "Graph Representation of Relation R2")

# Functions examples:

# a. Surjection (onto): f: {1,2,3} → {a,b}
surjection = {1: 'a', 2: 'a', 3: 'b'}
print("Surjection function:", surjection)

# b. Bijection (1-to-1 and onto): f: {1,2,3} → {a,b,c}
bijection = {1: 'a', 2: 'b', 3: 'c'}
print("Bijection function:", bijection)

# c. Many-to-one: f: {1,2,3} → {x,y}
many_to_one = {1: 'x', 2: 'x', 3: 'y'}
print("Many-to-one function:", many_to_one)

# d. Into function (injective but not surjective):
# f: {1,2,3} → {a,b,c,d}
into_function = {1: 'a', 2: 'b', 3: 'c'}
print("Into function:", into_function)
