import numpy as np
import math
import time
from math import comb  # nécessaire pour calculer les coeff binomiaux

# ----------------------------------------------------------------------------- 
# 1. Implementation récursive (with memoization)
# ----------------------------------------------------------------------------- 
def binomial_tree_recursive(S0, K, T, r, sigma, q, N, option_type='call', i=0, j=0, memo=None):
    """
    Recursively calculates the price of a European option (Call or Put) using the CRR model.
    Continuous dividends are incorporated via the drift (r - q).
    
      N          : nb étapes dans l'arbre
      option_type: 'call' out 'put'
      i, j       : Recursive indices (i = pas, j = nbre de ups)
      memo       : dictionnaire pour stocker les résultats déjà calculés

    Returns:
      Option price at node (i, j)
    """
    if memo is None:
        memo = {}

    # Si on est à la dernière étape, alors à l'étape N, calcculer le payoff et le prix du sous jacent 
    if i == N:
        dt = T / N # c pourquoi quand le T est petit dde base on a des divergences et résultats incohérents c pas la méthode appropriée forcément pour notre dataset
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        S = S0 * (u ** j) * (d ** (i - j))
        if option_type == 'call':
            return max(S - K, 0)
        elif option_type == 'put':
            return max(K - S, 0)
    
    key = (i, j)
    if key in memo:
        return memo[key]
    
    dt = T / N
    u = math.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    # calcul récursif pour les noeuds suivants
    value_up = binomial_tree_recursive(S0, K, T, r, sigma, q, N, option_type, i+1, j+1, memo)
    value_down = binomial_tree_recursive(S0, K, T, r, sigma, q, N, option_type, i+1, j, memo)
    
    value = np.exp(-r * dt) * (p * value_up + (1 - p) * value_down)
    memo[key] = value
    return value

# ----------------------------------------------------------------------------- 
# 2. implementation vectorisée
# ----------------------------------------------------------------------------- 
def binomial_tree_vectorized(S0, K, T, r, sigma, q, N, option_type='call'):
    """
    Calculate the price of a European option (Call or Put) using the vectorized binomial tree method.
    Continuous dividends are integrated via (r - q).
    
      N          : nbre étapes (par ex 80)
      option_type: 'call' or 'put'
      
    Returns:
      Option price.
    """
    dt = T / N # même remarque pour la divergence du dataset et mauvaise valorisation
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    # calculer les prix du sous jacent à chaque évolution
    j = np.arange(0, N + 1)
    S_T = S0 * (u ** j) * (d ** (N - j))
    
    if option_type == 'call':
        payoff = np.maximum(S_T - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    
    coeff = np.array([comb(N, i) for i in j])
    
    # on actualise pour calculer prix option
    option_value = np.exp(-r * T) * np.sum(coeff * (p ** j) * ((1 - p) ** (N - j)) * payoff)
    return option_value

# ----------------------------------------------------------------------------- 
# 3. Main Function with User Input
# ----------------------------------------------------------------------------- 
def main():
    # Input market parameters
    S0 = float(input("Enter the initial underlying asset price (S0): "))
    K = float(input("Enter the strike price (K): "))
    T = float(input("Enter the time to maturity in years (T): "))
    r = float(input("Enter the risk-free rate (r): "))
    sigma = float(input("Enter the volatility (sigma): "))
    q = float(input("Enter the continuous dividend yield (q): "))
    option_type = input("Enter the option type ('call' or 'put'): ").strip().lower()
    N = int(input("Enter the number of steps (e.g., 80): "))
    
    # temps récursif
    start_rec = time.time()
    price_rec = binomial_tree_recursive(S0, K, T, r, sigma, q, N, option_type)
    time_rec = time.time() - start_rec
    
    # temps itératif
    start_vec = time.time()
    price_vec = binomial_tree_vectorized(S0, K, T, r, sigma, q, N, option_type)
    time_vec = time.time() - start_vec
    
    print("\nResults:")
    print(f"Option Price ({option_type.capitalize()}) - Recursive: {price_rec:.4f} EUR (Time: {time_rec:.4f} sec)")
    print(f"Option Price ({option_type.capitalize()}) - Vectorized: {price_vec:.4f} EUR (Time: {time_vec:.4f} sec)")

if __name__ == "__main__":
    main()
