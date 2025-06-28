# Artificial Intelligence for Business
# Optimizing warehouse flows with Q-learning

# Importing the libraries
import numpy as np
import csv
from datetime import datetime

# Parameters gamma and alpha
gamma = 0.75
alpha = 0.9

# States
location_to_state = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11
}
state_to_location = {state: location for location, state in location_to_state.items()}

# Actions
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def save_route_to_csv(start_point, end_point, route, filename="optimal_routes.csv"):
    """Save the route information to a CSV file"""
    try:
        # Try to open the file in append mode first to check if it exists
        with open(filename, 'a', newline='') as file:
            pass
    except FileNotFoundError:
        # If file doesn't exist, create it with headers
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Start Point", "End Point", "Optimal Route", "Number of Steps"])

    # Append the new route information
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        route_str = " -> ".join(route)
        writer.writerow([timestamp, start_point, end_point, route_str, len(route)])

def route(starting_location, ending_location):
    # Rewards
    R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
                  [1,0,1,0,0,1,0,0,0,0,0,0],
                  [0,1,0,0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,1,0,0,0],
                  [0,1,0,0,0,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0,0,0],
                  [0,0,0,1,0,0,1,0,0,0,0,1],
                  [0,0,0,0,1,0,0,0,0,1,0,0],
                  [0,0,0,0,0,1,0,0,1,0,1,0],
                  [0,0,0,0,0,0,0,0,0,1,0,1],
                  [0,0,0,0,0,0,0,1,0,0,1,0]])

    ending_state = location_to_state[ending_location]
    R[ending_state, ending_state] = 1000

    # Initialisation des Q-values (training matrix Q)
    Q = np.zeros([12, 12])
    for _ in range(1000):
        # Nous exécutons une action aléatoire a_t
        current_state = np.random.randint(0, 12)
        # Liste des actions possibles
        playable_actions = []
        for j in range(12):
            if R[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

    # Find optimal route
    route_path = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[next_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route_path.append(next_location)

    # Save the route to CSV
    save_route_to_csv(starting_location, ending_location, route_path)

    return route_path


def route_priority(starting_location, ending_location, M=1000):
    # Rewards
    R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
                  [1,0,1,0,0,1,0,0,0,0,0,0],
                  [0,1,0,0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,1,0,0,0],
                  [0,1,0,0,0,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0,0,0],
                  [0,0,0,1,0,0,1,0,0,0,0,1],
                  [0,0,0,0,1,0,0,0,0,1,0,0],
                  [0,0,0,0,0,1,0,0,1,0,1,0],
                  [0,0,0,0,0,0,0,0,0,1,0,1],
                  [0,0,0,0,0,0,0,1,0,0,1,0]])

    ending_state = location_to_state[ending_location]
    R[ending_state, ending_state] = M
    
    for i, node_init in enumerate(nodes_init):
        R[node_init, node_init] = M-10*i

    # Initialisation des Q-values (training matrix Q)
    Q = np.zeros([12, 12])
    for _ in range(1000):
        # Nous exécutons une action aléatoire a_t
        current_state = np.random.randint(0, 12)
        # Liste des actions possibles
        playable_actions = []
        for j in range(12):
            if R[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

    # Find optimal route
    route_path = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[next_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route_path.append(next_location)

    # Save the route to CSV
    save_route_to_csv(starting_location, ending_location, route_path)

    return route_path

def best_route(starting_location, ending_location, intermediary_location):
    route1 = route(starting_location, intermediary_location)
    route2 = route(intermediary_location, ending_location)[1:]
    full_route = route1 + route2

    # Save the combined route to CSV
    save_route_to_csv(starting_location, ending_location, full_route)

    return full_route

# Interface utilisateur simple
print("Système d'optimisation de routes d'entrepôt")
print("----------------------------------------")
print("Options disponibles:")
print("1. Trouver la route optimale entre deux points")
print("2. Trouver la meilleure route via un point intermédiaire")
print("3. Quitter")

while True:
    choice = input("\nChoisissez une option (1-3): ")

    if choice == "1":
        start = input("Point de départ (A-L): ").upper()
        end = input("Point d'arrivée (A-L): ").upper()
        if start in location_to_state and end in location_to_state:
            optimal_route = route(start, end)
            print(f"\nRoute optimale: {' -> '.join(optimal_route)}")
            print(f"Le résultat a été enregistré dans optimal_routes.csv")
        else:
            print("Points invalides. Veuillez utiliser des lettres entre A et L.")

    elif choice == "2":
        start = input("Point de départ (A-L): ").upper()
        intermediary = input("Point intermédiaire (A-L): ").upper()
        end = input("Point d'arrivée (A-L): ").upper()
        if (start in location_to_state and intermediary in location_to_state
            and end in location_to_state):
            optimal_route = best_route(start, end, intermediary)
            print(f"\nMeilleure route: {' -> '.join(optimal_route)}")
            print(f"Le résultat a été enregistré dans optimal_routes.csv")
        else:
            print("Points invalides. Veuillez utiliser des lettres entre A et L.")

    elif choice == "3":
        print("Merci d'avoir utilisé notre système. Au revoir!")
        break

    else:
        print("Option invalide. Veuillez choisir 1, 2 ou 3.")