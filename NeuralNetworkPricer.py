import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import time

# charger le fichier et récupérer les colonnes pertinents
def load_and_preprocess_data(file_path, test_size=0.2):
    spoptions = pd.read_csv(file_path, delimiter=';')
    
    # input et output selon l'énoncé

    # o n peut entraineler le modèle sur les calls ou put en fonction des besoins de l'utilisateur

    inputs = spoptions[['UNDERLYING_LAST', 'STRIKE', 'Riskfreerate', 'P_IV', 'DTE', 'Dividendrate']].values
    outputs = spoptions['P_LAST'].values  # Taking the call option prices directly
    
    # diviser le dataset en train et test on fait évidemment un mélange aléatoire pour ne pas avoir de déséquilibre
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_size, random_state=42)
    
    # conversion aux tensors de pytorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# définir le neural network
class OptionPricerNN(nn.Module):
    """
    A neural network model for option pricing.
    """

    # on intialise les couches du réseau et créé la fonction d'activation en ReLu comme demandé
    def __init__(self): # on a choisi 6 couches car le nb de features est 6 mais on peut affiner -> hyperparamètre celui qui donne une rmse et temps d'exécution correcte via benchmarks
        super(OptionPricerNN, self).__init__()
        self.layer1 = nn.Linear(6, 256)  # Input features: S0, K, r, sigma, T, q y en a bien 6
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 32)
        self.layer6 = nn.Linear(32, 1)  # Output is the option price
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return x

# entraînement
def train_model(model, X_train_tensor, y_train_tensor, epochs=100, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # on en met un petit pour convergence lente mais sûre pour minimser le RMSE
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs_train = model(X_train_tensor)
        loss = loss_fn(outputs_train, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Scheduler step
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Step 4: Evaluate the Model
def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        # Predicting the test set
        outputs_test = model(X_test_tensor)
        
        # Calculate RMSE, MSE, MAE
        test_loss_mse = mean_squared_error(y_test_tensor.numpy(), outputs_test.numpy())
        test_loss_rmse = np.sqrt(test_loss_mse)
        test_loss_mae = mean_absolute_error(y_test_tensor.numpy(), outputs_test.numpy())
        
        print(f"Test RMSE: {test_loss_rmse:.4f}")
        print(f"Test MSE: {test_loss_mse:.4f}")
        print(f"Test MAE: {test_loss_mae:.4f}")

# Step 5: Save the Model
def save_model(model, filename='option_pricer_model.pth'):
    torch.save(model.state_dict(), filename)

# Main function to execute all steps

def programmePrincipal():

    debut_simulation = time.time()

    # Load and preprocess data
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data('spoptions_cleaned.csv')

        # Initialize the model
    model = OptionPricerNN()


        # Train the model
    train_model(model, X_train_tensor, y_train_tensor, epochs=100, learning_rate=0.001)

    fin_entrainement = time.time() - debut_simulation

    print(fin_entrainement)

    debut_test = time.time()
    # Evaluate the model
    evaluate_model(model, X_test_tensor, y_test_tensor)

    fin_test = time.time() - debut_test


    #    Save the trained model
    save_model(model)

    print(fin_test)



def main():
    programmePrincipal()

if __name__ == "__main__":
    main()
