import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# =============================
# 1. DADOS (2000–2030)
# =============================
anos = np.arange(2000, 2031).reshape(-1, 1)

vendas = np.array([
    10, 11, 11.5, 12, 13, 14, 14.5, 15, 16, 17, 18,
    19, 19.5, 20, 21, 22, 23.5, 24, 25.5, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37 ,
    sair
    
]).reshape(-1, 1)

# =============================
# 2. NORMALIZAÇÃO
# =============================
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

anos_norm = scaler_x.fit_transform(anos)
vendas_norm = scaler_y.fit_transform(vendas)

# =============================
# 3. TREINO / TESTE
# =============================
x_train, x_test, y_train, y_test = train_test_split(
    anos_norm, vendas_norm, test_size=0.2, random_state=42
)

# =============================
# 4. MODELO MLP
# =============================
modelo = MLPRegressor(
    hidden_layer_sizes=(50, 50),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

# =============================
# 5. TREINAMENTO
# =============================
modelo.fit(x_train, y_train.ravel())

# =============================
# 6. AVALIAÇÃO
# =============================
y_pred = modelo.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nErro Quadrático Médio (MSE): {mse:.6f}")

# =============================
# 7. INTERAÇÃO COM UTILIZADOR
# =============================
print("\n=== SISTEMA DE PREVISÃO DE VENDAS ===")

while True:
    entrada = input("\nDigite um ano para previsão (ou 'sair'): ")

    if entrada.lower() == "sair":
        print("Sistema encerrado.")
        break

    try:
        ano = int(entrada)

        ano_norm = scaler_x.transform([[ano]])
        previsao_norm = modelo.predict(ano_norm)
        previsao = scaler_y.inverse_transform(previsao_norm.reshape(-1, 1))

        print(f"👉 Previsão de vendas para {ano}: {previsao[0][0]:.2f}")

    except ValueError:
        print("⚠️ Entrada inválida. Digite um ano válido.")

# =============================
# 8. GRÁFICO FINAL
# =============================
anos_plot = scaler_x.inverse_transform(anos_norm)
vendas_prev_norm = modelo.predict(anos_norm)
vendas_prev = scaler_y.inverse_transform(vendas_prev_norm.reshape(-1, 1))

plt.figure(figsize=(10, 5))
plt.plot(anos_plot, vendas, label="Vendas Reais")
plt.plot(anos_plot, vendas_prev, label="Previsão MLP")
plt.xlabel("Ano")
plt.ylabel("Vendas")
plt.title("Vendas Reais vs Previsão (MLP)")
plt.legend()
plt.grid(True)
plt.show()

