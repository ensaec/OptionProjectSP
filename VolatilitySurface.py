# programme pour faire qq statistiques descriptives sur le dataframe et cleaner les données

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def afficherSurfaceVol():

    spoptions = pd.read_csv('spoptions_cleaned.csv', delimiter=';') # je spécifie le délimiteur pour pas avoir une seule colonne

    df_vol_surface = spoptions[['UNDERLYING_LAST', 'DTE', 'C_IV', 'STRIKE']]

    # Calculer la moneyness
    df_vol_surface['Moneyness'] = df_vol_surface['UNDERLYING_LAST'] / df_vol_surface['STRIKE']

    # Convertir DTE en années
    df_vol_surface['DTE'] = df_vol_surface['DTE'] / 365  # Diviser par 365 pour obtenir les années


    # Créer un maillage (meshgrid) pour la moneyness et le temps jusqu'à maturité

    # je veux des poitns de données uniques

    moneyness = np.unique(df_vol_surface['Moneyness'])
    maturities = np.unique(df_vol_surface['DTE'])

    # Créer une grille de points pour l'interpolation
    X, Y = np.meshgrid(moneyness, maturities)

    # Interpoler les valeurs de volatilité implicite
    Z = griddata(
        (df_vol_surface['Moneyness'], df_vol_surface['DTE']),
        df_vol_surface['C_IV'],
     (X, Y),
        method='cubic'  # Utilisation de l'interpolation bicubique
    )

    # Tracer la surface de volatilité
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Labels et titre avec unités explicites
    ax.set_xlabel('Moneyness (Underlying / Strike)')
    ax.set_ylabel('Time to Maturity (Years)')
    ax.set_zlabel('Implied Volatility (%)')
    ax.set_title('Volatility Surface')

    # Afficher le graphique
    plt.show()

def main():
    afficherSurfaceVol()
    
if __name__ == "__main__":
    afficherSurfaceVol()
