from BlackScholesPricer import black_scholes
from BlackScholesPricer import calculate_greeks
from BinomialTreePricer import binomial_tree_vectorized
from MonteCarloPricer import  simulate_final_prices
from MonteCarloPricer import option_price_and_payoffs
from MonteCarloPricer import compute_greeks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from NeuralNetworkPricer import OptionPricerNN, train_model, train_test_split  # assuming you already have OptionPricerNN and train_model in your existing module

# dans le fichier cleané on récupère tous les vecteurs de paramètres initiaux pour chaque option

spoptions = pd.read_csv('spoptions_cleaned.csv', delimiter=';') 

S0 = spoptions['UNDERLYING_LAST'].values
K = spoptions['STRIKE'].values
T = spoptions['DTE'].values
r = spoptions['Riskfreerate'].values
sigmaCall = spoptions['C_IV'].values
sigmaPut = spoptions['P_IV'].values
prixCall = spoptions['C_LAST'].values
prixPut = spoptions['P_LAST'].values
q = spoptions['Dividendrate'].values

def performanceBlackScholes():
    
    start_recording_BSM = time.time()

    

    bs_prices_calls = np.array([black_scholes(S0[i], K[i], T[i], r[i], sigmaCall[i], q[i], option_type='call') for i in range(len(S0))])
    bs_prices_puts = np.array([black_scholes(S0[i], K[i], T[i], r[i], sigmaPut[i], q[i], option_type='put') for i in range(len(S0))])

    # RMSE 
    rmse_bs_call = np.sqrt(mean_squared_error(bs_prices_calls, prixCall))

    rmse_bs_put = np.sqrt(mean_squared_error(bs_prices_puts, prixPut))

    # même si les options sont les mêmes, je préfère calculer le rmse en moyenne des 2 car il se trouve que les puts sont mieux estimés que les calls
    # c dû au fait que bcp d'options hors de la monnaie pour un call 

    mae_bs_call = mean_absolute_error(bs_prices_calls, prixCall)
    mae_bs_put = mean_absolute_error(bs_prices_puts, prixPut)
    # on a occulé le calcul et la comparaison des grecques ici même si on aurait pu la faire le code compile en 60s au lieu de 10s si on fait la comparaison

    print(rmse_bs_call)
    print(rmse_bs_put)

    print("RMSE moyen" + str((rmse_bs_call + rmse_bs_put)/2))

    print("MAE moyen" + str((mae_bs_call + mae_bs_put)/2))


    tempsexecution = time.time() - start_recording_BSM

    print("temps exécution pour pricer 263 000 options" + str(tempsexecution))

def performanceMonteCarlo():
    start_recording_MonteCarlo = time.time()

    # Initialize lists to store the results
    montecarlo_prices_calls = np.array([])
    montecarlo_prices_puts = np.array([])
    
    # Loop through each option in the dataset (row by row)
    for i in range(len(S0)):
        # Extract parameters for the current option
        S0_i = S0[i]
        K_i = K[i]
        T_i = T[i]
        r_i = r[i]
        sigmaCall_i = sigmaCall[i]
        sigmaPut_i = sigmaPut[i]
        q_i = q[i]
        

         # Check the shape of the results to ensure consistency
       
        
        # Run the Monte Carlo simulation for Call and Put options separately
        call_T_i, _, _ = simulate_final_prices(S0_i, T_i, r_i, sigmaCall_i, q_i, 1000, True) # trajectoire similaire pour un put et call
        # Calculate the option prices using Monte Carlo simulations
        montecarlo_price_call, _ = option_price_and_payoffs(S0_i, K_i, T_i, r_i, sigmaCall_i, q_i, option_type='call', S_T=call_T_i) # on prend le prix uniquement pas le payoff
        montecarlo_price_put, _ = option_price_and_payoffs(S0_i, K_i, T_i, r_i, sigmaPut_i, q_i, option_type='put', S_T=call_T_i) #idem

        # Append the results to the lists
        montecarlo_prices_calls = np.append(montecarlo_prices_calls, montecarlo_price_call)
        montecarlo_prices_puts = np.append(montecarlo_prices_puts, montecarlo_price_put)
    


    # Ensure no empty or inconsistent data is included
    if montecarlo_prices_calls.size == 0 or montecarlo_prices_puts.size == 0:
        print("No valid Monte Carlo prices for options.")
        return

    # Compute RMSE, MSE, and MAE for Calls and Puts
    rmse_montecarlo_call = np.sqrt(mean_squared_error(montecarlo_prices_calls, prixCall))
    rmse_montecarlo_put = np.sqrt(mean_squared_error(montecarlo_prices_puts, prixPut))
    
    mse_montecarlo_call = mean_squared_error(montecarlo_prices_calls, prixCall)
    mse_montecarlo_put = mean_squared_error(montecarlo_prices_puts, prixPut)
    
    mae_montecarlo_call = mean_absolute_error(montecarlo_prices_calls, prixCall)
    mae_montecarlo_put = mean_absolute_error(montecarlo_prices_puts, prixPut)

    # Output RMSE, MSE, and MAE values
    print(f"RMSE for Call options: {rmse_montecarlo_call:.4f}")
    print(f"RMSE for Put options: {rmse_montecarlo_put:.4f}")
    print(f"Average RMSE: {(rmse_montecarlo_call + rmse_montecarlo_put) / 2:.4f}")
    
    print(f"MSE for Call options: {mse_montecarlo_call:.4f}")
    print(f"MSE for Put options: {mse_montecarlo_put:.4f}")
    
    print(f"MAE for Call options: {mae_montecarlo_call:.4f}")
    print(f"MAE for Put options: {mae_montecarlo_put:.4f}")

    # Measure execution time
    tempsexecution = time.time() - start_recording_MonteCarlo
    print(f"Execution time for pricing {len(S0)} options: {tempsexecution:.2f} seconds")


def performanceBinomialTree():
    start_recording_BinomialTree = time.time()

    # Initialisation des listes pour stocker les résultats
    binomial_tree_prices_calls = []
    binomial_tree_prices_puts = []
    
    # Traitement des données incohérentes pour assurer un meilleur calcul des métriques de précision
    # Sinon, on a des résultats qui tendent vers l'infini
    tableauValeursIncoherentesCall = []
    tableauValeursIncoherentesPut = []
    vraiCall = []
    vraiPut = []

    # Boucle sur chaque option dans le dataset
    for i in range(len(S0)):
        # Extraction des paramètres pour l'option courante
        S0_i = S0[i]
        K_i = K[i]
        T_i = T[i]
        r_i = r[i]
        sigmaCall_i = sigmaCall[i]
        sigmaPut_i = sigmaPut[i]
        q_i = q[i]
        
        # Exécution du modèle binomial pour les options Call et Put
        call_price = binomial_tree_vectorized(S0_i, K_i, T_i, r_i, sigmaCall_i, q_i, N=10, option_type='call')
        put_price = binomial_tree_vectorized(S0_i, K_i, T_i, r_i, sigmaPut_i, q_i, N=10, option_type='put')
        
        # Calcul de la différence entre le prix calculé et le prix observé
        differenceCall = abs(call_price - prixCall[i]) 
        differencePut = abs(put_price - prixPut[i])

        # Si la différence dépasse un seuil, on considère la donnée comme incohérente
        if differencePut > 5000:
            tableauValeursIncoherentesPut.append(differencePut)
        else:
            # Ajout des résultats valides à la liste
            binomial_tree_prices_puts.append(put_price)
            vraiPut.append(prixPut[i])

        if differenceCall > 5000:
            tableauValeursIncoherentesCall.append(differenceCall)
        else:
            # Ajout des résultats valides à la liste
            binomial_tree_prices_calls.append(call_price)
            vraiCall.append(prixCall[i])
    
    # Conversion des listes en tableaux numpy pour faciliter les calculs
    binomial_tree_prices_calls = np.array(binomial_tree_prices_calls)
    binomial_tree_prices_puts = np.array(binomial_tree_prices_puts)

    # Affichage du nombre de résultats incohérents trouvés
    print(f"Nombre de résultats incohérents pour les Call: {len(tableauValeursIncoherentesCall)}")  # Environ 16k options, soit plus de 10% du dataset
    print(f"Nombre de résultats incohérents pour les Put: {len(tableauValeursIncoherentesPut)}")

    # Calcul de la RMSE pour les Call et Put
    rmse_binomial_call = np.sqrt(mean_squared_error(binomial_tree_prices_calls, vraiCall))
    rmse_binomial_put = np.sqrt(mean_squared_error(binomial_tree_prices_puts, vraiPut))

    mae_binomial_call = np.sqrt(mean_absolute_error(binomial_tree_prices_calls, vraiCall))  
    mae_binomial_put = np.sqrt(mean_absolute_error(binomial_tree_prices_puts, vraiPut))

    # Affichage des RMSE
    print(f"RMSE pour les options Call: {rmse_binomial_call:.4f}")
    print(f"RMSE pour les options Put: {rmse_binomial_put:.4f}")
    print(f"RMSE moyen: {(rmse_binomial_call + rmse_binomial_put) / 2:.4f}")

    print(f"MAE pour les options Call: {mae_binomial_call:.4f}")
    print(f"MAE pour les options Put: {mae_binomial_put:.4f}")  
    print(f"MAE moyen: {(mae_binomial_call + mae_binomial_put) / 2:.4f}")

    # Mesure du temps d'exécution
    tempsexecution = time.time() - start_recording_BinomialTree
    print(f"Temps d'exécution pour calculer les prix de 263 000 options: {tempsexecution:.2f} secondes")



def performanceNeuralNetwork():
    # Step 1: Load the preprocessed data
    spoptions = pd.read_csv('spoptions_cleaned.csv', delimiter=';')
    inputs = spoptions[['UNDERLYING_LAST', 'STRIKE', 'Riskfreerate', 'C_IV', 'DTE', 'Dividendrate']].values
    outputs = spoptions['C_LAST'].values  # Taking the call option prices directly
    
    # Splitting the dataset into training and test sets (using 90% for training, 10% for testing)
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.1, random_state=42)
    
    # Converting to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Step 2: Initialize and train the model
    model = OptionPricerNN()
    train_model(model, X_train_tensor, y_train_tensor, epochs=100, learning_rate=0.001)  # Training the model

    # Step 3: Evaluation on the test set
    model.eval()
    with torch.no_grad():
        # Predicting the test set
        outputs_test = model(X_test_tensor)
        
        # Calculating RMSE, MSE, and MAE
        test_loss_mse = mean_squared_error(y_test_tensor.numpy(), outputs_test.numpy())
        test_loss_rmse = np.sqrt(test_loss_mse)
        test_loss_mae = mean_absolute_error(y_test_tensor.numpy(), outputs_test.numpy())
        
        # Print the results
        print(f"Test RMSE: {test_loss_rmse:.4f}")
        print(f"Test MSE: {test_loss_mse:.4f}")
        print(f"Test MAE: {test_loss_mae:.4f}")


def main():
    performanceMonteCarlo()

  

if __name__ == "__main__":
    performanceBinomialTree()

