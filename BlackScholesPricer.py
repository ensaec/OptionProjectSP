import numpy as np
import scipy.stats as si
from scipy.stats import norm


# Fonction pour calculer le prix du Call ou Put selon Black-Scholes avec la formule standard
def black_scholes(S0, K, T, r, sigma, q, option_type='call'):
    """
    Calcule le prix d'une option européenne (Call ou Put) selon Black-Scholes avec dividendes constants.
    :param S0: Prix initial de l'actif sous-jacent
    :param K: Prix d'exercice
    :param T: Temps jusqu'à l'échéance (en années)
    :param r: Taux sans risque
    :param sigma: Volatilité (exprimée normalement pas en %)
    :param q: Dividende constant (idem)
    :param option_type: Type d'option ('call' ou 'put')
    :return: Prix de l'option
    """
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)) # formules connues
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S0 * np.exp(-q * T) * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S0 * np.exp(-q * T) * si.norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return option_price

# 2. Calcul des grecques les paramètres de marché sont les mêmes
def calculate_greeks(S0, K, T, r, sigma, q, option_type='call'):

    # Calcul de d1 et d2
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta : Sensibilité du prix de l'option par rapport au prix de l'actif sous-jacent
    if option_type == 'call':
        delta = np.exp(-q * T) * si.norm.cdf(d1)
    elif option_type == 'put':
        delta = np.exp(-q * T) * (si.norm.cdf(d1) - 1)
    
    # Gamma : Sensibilité de Delta par rapport au prix de l'actif sous-jacent
    gamma = np.exp(-q * T) * si.norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    
    # Theta : Sensibilité du prix de l'option au temps
    if option_type == 'call':
        theta = (-S0 * si.norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2) + q * S0 * np.exp(-q * T) * si.norm.cdf(d1)
    elif option_type == 'put':
        theta = (-S0 * si.norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2) - q * S0 * np.exp(-q * T) * si.norm.cdf(-d1)
    
    # Vega : Sensibilité du prix de l'option à la volatilité
    vega = S0 * np.sqrt(T) * np.exp(-q * T) * si.norm.pdf(d1)
    
    # Rho : Sensibilité du prix de l'option au taux sans risque
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == 'put':
        rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
    
    # Charm : Sensibilité de Delta au passage du temps (pour Call et Put)
    charm = -np.exp(-q * T) * si.norm.pdf(d1) * (2 * (r - q) * np.sqrt(T) - sigma) / (2 * T)
    
    # Veta : Sensibilité de Vega au passage du temps (pour Call et Put)
    veta = -np.exp(-q * T) * si.norm.pdf(d1) * np.sqrt(T) * (d1 * d2) / (2 * T)
    
    # Retourner toutes les grecques
    greeks = {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho,
        'charm': charm,
        'veta': veta
    }
    
    return greeks

# fonction nécessaire pour que mon dataFrame soit complet
# Fonction pour obtenir q à partir du prix de l'option
# Fonction de dichotomie pour trouver le taux de dividende
# nous n'utiliserons pas cette fonction car les données sont mal calibrées et résultats incohérents (dividendes égaux à 1 car maturité très faible)
# j'ai préféré remplir manuellement avec les données historiques
def find_dividend_rate(S, K, T, r, sigma, observed_option_price, option_type, tolerance=1e-4, max_iter=50):
    # Initialisation de l'intervalle de recherche
    q_min = 0.0    # Limite inférieure
    q_max = 1.0    # Limite supérieure
    iteration = 0

    while iteration < max_iter:
        # Calcul du point médian
        q_mid = (q_min + q_max) / 2
        
        # Calcul du prix de l'option avec le taux de dividende actuel (q_mid)
        calculated_option_price = black_scholes(S, K, T, r, sigma, q_mid, option_type)
        
        # Vérification de la différence entre le prix calculé et le prix observé
        price_diff = calculated_option_price - observed_option_price
        
        # Si la différence est assez petite, on a trouvé notre solution
        if abs(price_diff) < tolerance:
            return q_mid
        
        # Réduction de l'intervalle de recherche
        if price_diff > 0:
            q_max = q_mid  # Le taux de dividende doit être plus petit
        else:
            q_min = q_mid  # Le taux de dividende doit être plus grand
        
        iteration += 1
    
    # Si la convergence échoue, retourner la meilleure estimation
    return (q_min + q_max) / 2


# 4. Fonction principale pour demander les entrées de l'utilisateur et afficher les résultats
def main():
    # Demander à l'utilisateur s'il veut un Call ou un Put
    option_choice = input("Voulez-vous calculer un Call ou un Put ? (Entrez 'call' ou 'put'): ").strip().lower()

    while option_choice != 'put' and option_choice != 'call':  # Correction de l'opérateur logique
        option_choice = input("Entrée invalide. Veuillez entrer 'call' ou 'put'.").strip().lower()
    
    # Demander les données de marché à l'utilisateur
    S0 = float(input("Entrez le prix initial de l'actif sous-jacent (S0) : "))
    K = float(input("Entrez le prix d'exercice (K) : "))
    T = float(input("Entrez le temps jusqu'à l'échéance en années (T) : "))
    r = float(input("Entrez le taux sans risque (r) (ex pour 3% mettez 0.03)  : "))
    sigma = float(input("Entrez la volatilité (sigma) (ex pour 5% mettez 0.05): "))
    q = float(input("Entrez le taux de dividende constant (q) (ex pour 4% mettez 0.04) : "))

    # Calcul du prix de l'option (Call ou Put)
    option_price = black_scholes(S0, K, T, r, sigma, q, option_choice)
    
    print("\n")  # Ajout d'une ligne vide pour la lisibilité

    # Affichage du prix de l'option
    print(f"Prix de l'option {option_choice.capitalize()} : {option_price:.2f} EUR")
    
    # Calcul des grecques
    greeks = calculate_greeks(S0, K, T, r, sigma, q, option_choice)
    
    print("\n")  # Ajout d'une ligne vide pour la lisibilité

    # Affichage des grecques
    print("\nGrecques de l'option :")
    print(f"Delta : {greeks['delta']:.4f}")
    print(f"Gamma : {greeks['gamma']:.4f}")
    print(f"Theta : {greeks['theta']:.4f}")
    print(f"Vega : {greeks['vega']:.4f}")
    print(f"Rho : {greeks['rho']:.4f}")
    print(f"Charm : {greeks['charm']:.4f}")
    print(f"Veta : {greeks['veta']:.4f}")

if __name__ == "__main__":
    main()
