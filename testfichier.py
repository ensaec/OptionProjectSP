# programme pour faire qq statistiques descriptives sur le dataframe et cleaner les données
#inutile pour la suite je le garde pour donner la démarche intellectuelle


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

spoptions = pd.read_csv('dataFrameSP.csv', delimiter=';') # je spécifie le délimiteur pour pas avoir une seule colonne

print(spoptions.head(6)) # stats des


print(spoptions.tail(7)) #idem

print(spoptions.dtypes) # on remarque que ya bcp de objets qu'il faudra convertir

# Liste des colonnes spécifiques à convertir
columns_to_convert = ['DTE', 'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO', 'C_IV', 'STRIKE', 
                      'P_DELTA', 'P_GAMMA', 'P_VEGA', 'P_THETA', 'P_RHO', 'P_IV', 'STRIKE_DISTANCE', 'Riskrate', 'Dividendrate']

# Conversion des colonnes spécifiques en flottants
spoptions[columns_to_convert] = spoptions[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Vérification des types de données après conversion
print(spoptions.dtypes)

spoptions = spoptions.dropna(subset=['C_IV', 'P_IV'])

print(spoptions.tail(10))

# Extraire les colonnes pertinentes et ajuster DTE en années
df_vol_surface = spoptions[['UNDERLYING_LAST', 'DTE', 'C_IV', 'STRIKE']]

# Convertir DTE en années
df_vol_surface['DTE'] = df_vol_surface['DTE'] / 365  # Diviser par 365 pour obtenir les années

# Assurer que la volatilité est en décimales (par exemple 25% devient 0.25)
df_vol_surface['C_IV'] = df_vol_surface['C_IV']

# Créer un maillage (meshgrid) pour les strikes et le temps jusqu'à maturité
strikes = np.unique(df_vol_surface['STRIKE'])
maturities = np.unique(df_vol_surface['DTE'])

# Créer une grille de points pour l'interpolation
X, Y = np.meshgrid(strikes, maturities)

# Interpoler les valeurs de volatilité implicite
Z = griddata(
    (df_vol_surface['STRIKE'], df_vol_surface['DTE']),
    df_vol_surface['C_IV'],
    (X, Y),
    method='cubic'  # Utilisation de l'interpolation bicubique
)

# Tracer la surface de volatilité
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Labels et titre avec unités explicites
ax.set_xlabel('Strike (USD)')
ax.set_ylabel('Time to Maturity (Years)')
ax.set_zlabel('Implied Volatility (Decimal)')
ax.set_title('Volatility Surface')

# Afficher le graphique
plt.show()