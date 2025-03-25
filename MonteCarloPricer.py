import numpy as np
import scipy.stats as si
import time

# cette fonction marche à la fois pour les calls et puts car simule le prix de l'action pas option en soit on verra ça après

def simulate_final_prices(S0, T, r, sigma, q, M, use_antithetic=True): # on utilise antithéthique pour diviser le nb de simus par 2 tout en gardant en précision

    """
    Simule les prix finaux S_T d'une option européenne en une seule étape.
    :param S0: Prix initial
    :param T: Temps à l'échéance
    :param r: Taux sans risque
    :param sigma: Volatilité
    :param q: Taux de dividende constant
    :param M: Nombre de simulations
    :param use_antithetic: Si True, utilise les variables antithétiques
    :return: Vecteur numpy de S_T et le vecteur Z utilisé
    """
    if use_antithetic:
        M_half = int(M/2)
        Z = np.random.randn(M_half)
        Z = np.concatenate([Z, -Z])
    else:
        Z = np.random.randn(M)
    
    # Simulation vectorisée du prix final (formule exacte pour European)
    X = np.exp((r - q - 0.5 * sigma**2)*T + sigma * np.sqrt(T) * Z)
    S_T = S0 * X
    return S_T, Z, X

def option_price_and_payoffs(S0, K, T, r, sigma, q, option_type, S_T):
    """
   
    :param option_type: 'call' ou 'put'
    :param S_T: Vecteur des prix finaux simulés qu'on a simulé dans la première fonction
    :return: Prix de l'option et vecteur des payoffs
    """

    # j'utilise les formules classiques du call et put
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    price = np.exp(-r*T) * np.mean(payoffs)
    return price, payoffs

def compute_greeks(S0, K, T, r, sigma, q, option_type, M, use_antithetic, base_Z, base_X):
    """
    Calcule Delta, Gamma, Theta, Vega et Rho en réévaluant la formule du prix avec perturbations,
    en utilisant le même vecteur de tirages base.
    Les perturbations sont appliquées sur S0, T, sigma et r respectivement.
    :return: Dictionnaire des grecques estimées.
    """
    epsilon = 0.01  # Perturbation relative

    # le vecteur de tirages base peut être considéré comme random matrix  pour calcul des grecques
    
    # On conserve le même vecteur de tirages base : base_Z et base_X (X = exp((r - q -0.5 sigma^2)T + sigma sqrt(T)Z))
    # Pour le Call, le payoff est max(S0*X - K, 0) ; pour le Put, max(K - S0*X, 0)

    # --- Delta via méthode pathwise ---
    # Pour un call, delta_i = I[S0*X > K]*(S0*X/S0) = I[S0*X > K]*X.
    # Pour un put, delta_i = I[S0*X < K]*(-X).
    if option_type == 'call':
        indicator = (S0 * base_X > K).astype(float)
        delta_paths = indicator * base_X
    else:  # put
        indicator = (S0 * base_X < K).astype(float)
        delta_paths = -indicator * base_X
    delta = np.exp(-r*T) * np.mean(delta_paths)

    # --- Gamma par différences finies sur S0 ---
    S0_up = S0 * (1 + epsilon)
    S0_down = S0 * (1 - epsilon)
    # On utilise le même X que base_X pour réévaluer les trajectoires :
    if option_type == 'call':
        indicator_up = (S0_up * base_X > K).astype(float)
        delta_up = np.exp(-r*T) * np.mean(indicator_up * base_X)
        indicator_down = (S0_down * base_X > K).astype(float)
        delta_down = np.exp(-r*T) * np.mean(indicator_down * base_X)
    else:
        indicator_up = (S0_up * base_X < K).astype(float)
        delta_up = np.exp(-r*T) * np.mean(-indicator_up * base_X)
        indicator_down = (S0_down * base_X < K).astype(float)
        delta_down = np.exp(-r*T) * np.mean(-indicator_down * base_X)
    gamma = (delta_up - delta_down) / (2 * epsilon * S0)

    # --- Theta par différences finies sur T ---
    T_up = T + epsilon
    T_down = T - epsilon
    # Pour T perturbé, recalculer X en gardant le même Z :
    X_T_up = np.exp((r - q - 0.5*sigma**2)*T_up + sigma*np.sqrt(T_up)*base_Z)
    X_T_down = np.exp((r - q - 0.5*sigma**2)*T_down + sigma*np.sqrt(T_down)*base_Z)
    S_T_up = S0 * X_T_up
    S_T_down = S0 * X_T_down
    if option_type == 'call':
        payoffs_T_up = np.maximum(S_T_up - K, 0)
        payoffs_T_down = np.maximum(S_T_down - K, 0)
    else:
        payoffs_T_up = np.maximum(K - S_T_up, 0)
        payoffs_T_down = np.maximum(K - S_T_down, 0)
    price_T_up = np.exp(-r*T_up) * np.mean(payoffs_T_up)
    price_T_down = np.exp(-r*T_down) * np.mean(payoffs_T_down)
    theta = (price_T_up - price_T_down) / (2 * epsilon)

    # --- Vega par différences finies sur sigma ---
    sigma_up = sigma * (1 + epsilon)
    sigma_down = sigma * (1 - epsilon)
    X_sigma_up = np.exp((r - q - 0.5*sigma_up**2)*T + sigma_up*np.sqrt(T)*base_Z)
    X_sigma_down = np.exp((r - q - 0.5*sigma_down**2)*T + sigma_down*np.sqrt(T)*base_Z)
    S_T_sigma_up = S0 * X_sigma_up
    S_T_sigma_down = S0 * X_sigma_down
    if option_type == 'call':
        payoffs_sigma_up = np.maximum(S_T_sigma_up - K, 0)
        payoffs_sigma_down = np.maximum(S_T_sigma_down - K, 0)
    else:
        payoffs_sigma_up = np.maximum(K - S_T_sigma_up, 0)
        payoffs_sigma_down = np.maximum(K - S_T_sigma_down, 0)
    price_sigma_up = np.exp(-r*T) * np.mean(payoffs_sigma_up)
    price_sigma_down = np.exp(-r*T) * np.mean(payoffs_sigma_down)
    vega = (price_sigma_up - price_sigma_down) / (2 * epsilon * sigma)

    # --- Rho par différences finies sur r ---
    r_up = r + epsilon
    r_down = r - epsilon
    X_r_up = np.exp((r_up - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*base_Z)
    X_r_down = np.exp((r_down - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*base_Z)
    S_T_r_up = S0 * X_r_up
    S_T_r_down = S0 * X_r_down
    if option_type == 'call':
        payoffs_r_up = np.maximum(S_T_r_up - K, 0)
        payoffs_r_down = np.maximum(S_T_r_down - K, 0)
    else:
        payoffs_r_up = np.maximum(K - S_T_r_up, 0)
        payoffs_r_down = np.maximum(K - S_T_r_down, 0)
    price_r_up = np.exp(-r_up*T) * np.mean(payoffs_r_up)
    price_r_down = np.exp(-r_down*T) * np.mean(payoffs_r_down)
    rho = (price_r_up - price_r_down) / (2 * epsilon)
    
    return {
        'price': np.exp(-r*T) * np.mean(np.maximum((S0 * base_X - K) if option_type=='call' else (K - S0 * base_X), 0)),
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def main():
    # Entrées de l'utilisateur
    S0 = float(input("Entrez le prix initial de l'actif sous-jacent (S0) : "))
    K = float(input("Entrez le prix d'exercice (K) : "))
    T = float(input("Entrez le temps jusqu'à l'échéance en années (T) : "))
    r = float(input("Entrez le taux sans risque (r) : "))
    sigma = float(input("Entrez la volatilité (sigma) : "))
    q = float(input("Entrez le taux de dividende constant (q) : "))
    option_type = input("Entrez le type d'option ('call' ou 'put') : ").strip().lower()
    
    M = 10000  # Nombre de simulations
    use_antithetic = True  # Utiliser variables antithétiques
    
    # Mesurer le temps de calcul
    start_time = time.time()
    
    # Simulation unique pour tous les calculs
    S_T, Z, base_X = simulate_final_prices(S0, T, r, sigma, q, M, use_antithetic)
    base_price, _ = option_price_and_payoffs(S0, K, T, r, sigma, q, option_type, S_T)
    
    # Calcul des grecques sur la base du même vecteur de tirages
    greeks = compute_greeks(S0, K, T, r, sigma, q, option_type, M, use_antithetic, Z, base_X)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(base_price)

    # Affichage des résultats
    print(f"\nPrix de l'option : {greeks['price']:.4f} EUR")
    print(f"Delta : {greeks['delta']:.4f}")
    print(f"Gamma : {greeks['gamma']:.4f}")
    print(f"Theta : {greeks['theta']:.4f}")
    print(f"Vega : {greeks['vega']:.4f}")
    print(f"Rho : {greeks['rho']:.4f}")
    print(f"Temps de calcul : {elapsed:.4f} secondes")

if __name__ == "__main__":
    main()
