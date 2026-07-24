import numpy as np
import control as ct
import matplotlib.pyplot as plt

# 1. Definindo as matrizes do sistema original (Conversor CC-CC)
A_orig = np.array([[0, -83.33],
                   [500, -10]])
# Corrigindo as dimensões de B_orig, C_orig e D_orig
# B_orig para ser 2x1
# C_orig para ser 1x2
# D_orig para ser 1x1 (assumindo feedthrough direto nulo)
B_orig = np.array([[166.67],
                   [0]])
C_orig = np.array([[0, 1]])
D_orig = np.array([[0]])

# 2. Obtendo a Função de Transferência para extrair os coeficientes
sys_orig = ct.ss(A_orig, B_orig, C_orig, D_orig)
tf_sys = ct.tf(sys_orig)
print("Função de Transferência da Planta:")
print(tf_sys)

# A equação característica do sistema aberto é s^2 + 10s + 41665
# O numerador é 83335

# 3. Modelando o sistema na Forma de Variáveis de Fase (conforme exigido na alínea 'b' do Problema 37)
# Para a forma de variáveis de fase (forma canônica de controle):
# A = [[0, 1], [-a0, -a1]]
# B = [[0], [1]]
# C = [[b0, b1]] (onde N(s) = b1*s + b0)
# Dado D(s) = s^2 + 10s + 41665 => a1 = 10, a0 = 41665
# Dado N(s) = 83335 => b1 = 0, b0 = 83335
A_fase = np.array([[0, 1],
                   [-41665, -10]])
B_fase = np.array([[0],
                   [1]])
C_fase = np.array([[83335, 0]])

sys_fase = ct.ss(A_fase, B_fase, C_fase, D_orig)

# 4. Requisitos de Desempenho (alínea 'c')
OS = 0.20 # Ultrapassagem de 20%
Ts = 0.5  # Tempo de acomodação de 0.5s

# Calculando Zeta e Wn
zeta = -np.log(OS) / np.sqrt(np.pi**2 + np.log(OS)**2)
wn = 4 / (zeta * Ts)

print(f"\nZeta desejado: {zeta:.4f}")
print(f"Frequência Natural (Wn) desejada: {wn:.4f} rad/s")

# 5. Calculando os Polos Desejados em Malha Fechada
p1 = -zeta*wn + 1j * wn * np.sqrt(1 - zeta**2)
p2 = -zeta*wn - 1j * wn * np.sqrt(1 - zeta**2)
polos_desejados = [p1, p2]
print(f"\nPolos desejados: {polos_desejados[0]:.2f}, {polos_desejados[1]:.2f}")

# 6. Cálculo dos Ganhos K (Alocação de Polos - Fórmula de Ackermann)
K = ct.acker(A_fase, B_fase, polos_desejados)
print(f"\nMatriz de Ganhos de Realimentação K: {K}")

# 7. Criando o Sistema em Malha Fechada e Simulando (alínea 'f')
# K é um vetor 1D, para a multiplicação matricial B_fase @ K, K precisa ser um vetor linha 2D (1, n)
A_cl = A_fase - B_fase @ K[np.newaxis, :]
sys_cl = ct.ss(A_cl, B_fase, C_fase, D_orig)

# Simulação da resposta ao degrau
tempo, saida = ct.step_response(sys_cl)

# Plotando o resultado gráfico
plt.figure(figsize=(10, 6))
plt.plot(tempo, saida, label='Tensão de Saída $u_C(t)$', color='darkblue', linewidth=2)
plt.axhline(y=saida[-1], color='red', linestyle='--', label='Regime Permanente')
plt.title('Simulação Computacional - Resposta ao Degrau (Conversor CC-CC)')
plt.xlabel('Tempo (segundos)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()