# programme pour faire qq statistiques descriptives sur le dataframe et cleaner les données

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from RiskFreeCurve import get_treasury_rates
from RiskFreeCurve import interpolate_rate_curve
from RiskFreeCurve import get_interpolated_rate
from BlackScholesPricer import find_dividend_rate
from BlackScholesPricer import black_scholes
from datetime import datetime

# Dividendes historiques du S&P500 et leur période
historical_dividends = {
    "17/03/2023": 1.506204,
    "16/06/2023": 1.638367,
    "15/12/2023": 1.906073,
    "15/03/2024": 1.594937,
    "21/06/2024": 1.759024,
    "20/09/2024": 1.745531,
    "20/12/2024": 1.965548,
    "21/03/2025": 1.695528
}

# Estimations futures des dividendes pour les années suivantes
future_dividends = {
    datetime.strptime("16/06/2025", "%d/%m/%Y"): 1.82150,
    datetime.strptime("15/12/2025", "%d/%m/%Y"): 1.7252,
    datetime.strptime("15/03/2026", "%d/%m/%Y"): 2.2
}

# Convertir les dates en objets datetime (pour les historiques)
historical_dividends = {datetime.strptime(date, "%d/%m/%Y"): value for date, value in historical_dividends.items()}

# Fonction pour estimer les dividendes pour chaque maturité

# Fonction pour estimer les dividendes pour chaque maturité
def estimate_dividend(maturity_date):
    # Convertir la date de maturité en objet datetime (format jour/mois/année)
    maturity_date = datetime.strptime(maturity_date, "%d/%m/%Y")  # Utilisation du bon format de date
    
    # Si la maturité est avant la date du 17 mars 2023, le dividende est 0
    if maturity_date < datetime.strptime("17/03/2023", "%d/%m/%Y"):
        return 0
    
    # Si la maturité est après 2025, utiliser les dividendes futurs estimés
    if maturity_date >= datetime.strptime("15/03/2026", "%d/%m/%Y"):
        # Trouver la clé la plus proche dans future_dividends
        future_keys = sorted(future_dividends.keys())
        for key in future_keys:
            if maturity_date <= key:
                return future_dividends[key]
    
    # Calcul de la moyenne des dividendes historiques
    dividend_values = [dividend for date, dividend in historical_dividends.items() if date <= maturity_date]
    
    # Calcul de la moyenne simple des dividendes historiques
    if dividend_values:
        return np.mean(dividend_values)
    else:
        return 0

spoptions = pd.read_csv('testData.csv', delimiter=';') # je spécifie le délimiteur pour pas avoir une seule colonne

print(spoptions.head(6)) # stats descriptives


print(spoptions.tail(7)) #idem

print(spoptions.dtypes) # on remarque que ya bcp de objets qu'il faudra convertir

# Liste des colonnes spécifiques à convertir
columns_to_convert = ['DTE', 'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO', 'C_IV', 'STRIKE', 
                      'P_DELTA', 'P_GAMMA', 'P_VEGA', 'P_THETA', 'P_RHO', 'P_IV', 'STRIKE_DISTANCE', 'Riskfreerate', 'Dividendrate', 'P_LAST', 'C_LAST']

# Conversion des colonnes spécifiques en flottants
spoptions[columns_to_convert] = spoptions[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Supprimer les espaces inutiles dans les colonnes de dates parce que j'avais un espace dans chaque point de donnée ce qui causait un bug de conversion
spoptions['QUOTE_DATE'] = spoptions['QUOTE_DATE'].str.strip()
spoptions['EXPIRE_DATE'] = spoptions['EXPIRE_DATE'].str.strip()

# Convertir les colonnes en datetime en utilisant pd.to_datetime
spoptions['QUOTE_DATE'] = pd.to_datetime(spoptions['QUOTE_DATE'], errors='coerce')  # 'coerce' convertit les erreurs en NaT
spoptions['EXPIRE_DATE'] = pd.to_datetime(spoptions['EXPIRE_DATE'], errors='coerce')  # Idem

# Vérifier les résultats après conversion
print(spoptions[['QUOTE_DATE', 'EXPIRE_DATE']].head())

spoptions['Riskfreerate'] = spoptions['Riskfreerate'].astype(float)
spoptions['Dividendrate'] = spoptions['Dividendrate'].astype(float)

# Vérification des types de données après conversion

# Supprimer les lignes où les colonnes 'C_IV', 'P_IV', ou 'DTE' contiennent un 0
spoptions = spoptions[(spoptions['C_IV'] != 0) & (spoptions['P_IV'] != 0) & (spoptions['DTE'] != 0)]  # ça sert à rien d'évaluer des options qui périment le jour même ou quand on a pas de vol implicite pour le put ou le call
spoptions = spoptions.dropna(subset=['C_IV', 'P_IV', 'DTE'])

spoptions['DTE'] = spoptions['DTE'] / 365  # Diviser par 365 pour obtenir les années


print(spoptions.dtypes)

date_valorisation = "15/03/2023" # on se place au mois de mars comme convenu. Par souci de clarté et efficacité, on est obligé de prendre qu'une seule courbe de taux pour tout le mois de mars pour faire l'interpolation

# on est ddans le feature engineering où je vais compléter la colonne riskfreerate


maturities, rates = get_treasury_rates(date_valorisation)


    # Interpoler la courbe des taux
cubic_spline, _, _ = interpolate_rate_curve(maturities, rates)

i = 0 

    # 4. Mettre à jour la colonne 'Riskfreerate' et Dividendrate avec le taux interpolé pour chaque ligne et ALGORITHme de dichotomie
    #finalement, l'algorithme de dichotomie ne donne pas de bons résultats, je vais inputer manuellement
    # on a des paramètres de marché pas forcément les mieux calibrés mais pas grave


for index, row in spoptions.iterrows():
    maturity_years = row['DTE']  # Maturité en années

    # Interpolation du taux sans risque (Riskfreerate)
    interpolated_rate = get_interpolated_rate(cubic_spline, maturity_years)
    spoptions.at[index, 'Riskfreerate'] = interpolated_rate / 100

    # Récupérer la date d'expiration et la convertir en chaîne de caractères
    curDate = row['EXPIRE_DATE']
    
    # Vérifier si la date est déjà un Timestamp et la convertir en chaîne
    if isinstance(curDate, pd.Timestamp):
        curDate = curDate.strftime('%d/%m/%Y')  # Conversion en chaîne de caractères au format 'jour/mois/année'

    # Calculer le taux de dividende estimé et mettre à jour la colonne 'Dividendrate'
    spoptions.at[index, 'Dividendrate'] = estimate_dividend(curDate) / 100



"""""
irrelevant mais bon de garder pour epxliquer la démarche

        # Calcul du taux de dividende en récupérant les données historiques
        S = row['UNDERLYING_LAST']
        K = row['STRIKE']
        T = row['DTE'] 
        r = row['Riskfreerate']  # Taux sans risque
        sigma = 0 # Volatilité implicite
        observed_call_price = row['C_LAST']  # Prix observé de l'option call
        observed_put_price = row['P_LAST'] # prix observé pour le put

        prixASaisir = 0
        chaineTexte = ""
        if(observed_call_price == 0):
                prixASaisir = observed_put_price
                chaineTexte = "put"
                sigma = row['P_IV']
        else:
                prixASaisir = observed_call_price
                chaineTexte = "call"
                sigma = row['C_IV']
        
   dividend_rate = find_dividend_rate(S, K, T, r, sigma, prixASaisir, option_type=chaineTexte)             
"""

        
        



print(spoptions[['QUOTE_DATE', 'DTE', 'Riskfreerate', 'Dividendrate']].head(5))

# Exporter le DataFrame dans un fichier CSV
spoptions.to_csv('spoptions_cleaned.csv', index=False, sep=';')  # Délimiteur ';' pour le CSV

