import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
from datetime import datetime

def get_treasury_rates(valuation_date):
    """
    Récupère les taux des obligations du Trésor américain à différentes maturités pour le mois spécifié.
    Les symboles FRED sont utilisés pour accéder aux séries temporelles des taux des obligations du Trésor américain.
    """
    # Taux des obligations du Trésor US à différentes maturités
    treasury_symbols = {
        '1M': 'DTB4WK',   # Taux à 1 mois
        '3M': 'TB3MS',     # Taux à 3 mois
        '6M': 'DTB6',      # Taux à 6 mois
        '1Y': 'GS1',       # Taux à 1 an (1-Year Treasury Constant Maturity Rate)
        '2Y': 'GS2',       # Taux à 2 ans
        '3Y': 'GS3',       # Taux à 3 ans
        '5Y': 'GS5',       # Taux à 5 ans
        '7Y': 'GS7',       # Taux à 7 ans
        '10Y': 'GS10',     # Taux à 10 ans
        '20Y': 'GS20',     # Taux à 20 ans
        '30Y': 'GS30'      # Taux à 30 ans
    }

    rates = []
    maturities = []

    # Extraire le mois de la date de valorisation
    valuation_month = datetime.strptime(valuation_date, "%d/%m/%Y").month
    valuation_year = datetime.strptime(valuation_date, "%d/%m/%Y").year
    
    # Définir le début et la fin du mois de valorisation
    start_date = f'{valuation_year}-{valuation_month:02d}-01'
    end_date = f'{valuation_year}-{valuation_month:02d}-28'  # Assumer que le mois a 28 jours (pour simplification)

    for maturity, symbol in treasury_symbols.items():
        try:
            # Télécharger les données depuis FRED pour le mois spécifié
            data = web.DataReader(symbol, 'fred', start=start_date, end=end_date)
            
            # Vérifier si les données sont disponibles
            if data.empty:
                print(f"Pas de données disponibles pour {maturity} ({symbol}) dans le mois {valuation_month}/{valuation_year}.")
                continue
            
            # Récupérer le taux de rendement le plus récent du mois
            rate = data[symbol].iloc[-1]  # Dernier taux disponible

            # Conversion des maturités en fractions d'année
            if maturity == '1M':
                maturity_in_years = 1 / 12 # 1 mois = 1/12 d'année
            elif maturity == '3M':
                maturity_in_years = 0.25 # 1 quart d'année
            elif maturity == '6M':
                maturity_in_years = 0.5 # 1 moitié d'année
            else:
                maturity_in_years = int(maturity[:-1])  # Pour les autres maturités comme '1Y', '2Y', etc.
            
            maturities.append(maturity_in_years)
            rates.append(rate)

        except Exception as e:
            print(f"Erreur lors de la récupération des données pour {maturity} ({symbol}): {e}")
            continue

    return maturities, rates

def interpolate_rate_curve(maturities, rates):
    """
    Effectue une interpolation de la courbe des taux sans risque à partir des taux des obligations.
    Utilise l'interpolation spline cubique pour estimer la courbe des taux.
    """
    cubic_spline = CubicSpline(maturities, rates, bc_type='natural')

    # Générer les maturités pour l'axe des x
    x_new = np.linspace(0, 50, 5000)
    y_new = cubic_spline(x_new)

    return cubic_spline, x_new, y_new

def get_interpolated_rate(cubic_spline, maturity):
    """
    Retourne le taux interpolé pour une maturité donnée.
    """
    return cubic_spline(maturity)

def plot_rate_curve(x_new, y_new):
    """
    Trace la courbe des taux sans risque à l'aide de matplotlib.
    """
    plt.figure(figsize=(11, 7))
    plt.plot(x_new, y_new, label="Courbe des Taux Sans Risque", color='green')
    plt.title("Courbe des Taux Sans Risque (US Treasury Bonds)")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux (% annuel)")
    plt.grid(True)
    plt.legend()
    plt.show()

def main(valuation_date, maturity_years):
    # 1. Extraire l'année de la date de valorisation
    valuation_year = datetime.strptime(valuation_date, "%d/%m/%Y").year
    
    # 2. Récupérer les taux des obligations du Trésor pour le mois spécifié
    maturities, rates = get_treasury_rates(valuation_date)
    
    # Si aucune donnée valide n'a été récupérée
    if not maturities:
        print("Aucune donnée n'a été récupérée. Le programme va s'arrêter.")
        return
    
    # 3. Interpoler la courbe des taux
    cubic_spline, x_new, y_new = interpolate_rate_curve(maturities, rates)
    
    # 4. Obtenir le taux interpolé pour la maturité souhaitée
    interpolated_rate = get_interpolated_rate(cubic_spline, maturity_years)
    print(f"Taux interpolé pour {maturity_years} années : {interpolated_rate:.2f}%")
    
    # 5. Tracer la courbe des taux
    plot_rate_curve(x_new, y_new)


# Exemple d'appel de la fonction
if __name__ == "__main__":
    valuation_date = "15/01/2023"  # Date de valorisation que l'utilisateur doit saisir
    maturity_years = 0.3  # Temps de maturité souhaité en années
    main(valuation_date, maturity_years)
