#!/usr/bin/env python3
"""
Градиентный спуск для линейной регрессии
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def generate_data(n_samples: int = 1000, noise: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Генерирует синтетические данные для линейной регрессии y = 3x + 7 + noise"""
    np.random.seed(42)
    X = np.random.randn(n_samples, 1) * 10
    y = 3 * X + 7 + np.random.randn(n_samples, 1) * noise
    return X, y


def compute_loss(X: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """Mean Squared Error"""
    n = len(y)
    predictions = w * X + b
    loss = (1 / (2 * n)) * np.sum((predictions - y) ** 2)
    return loss


def compute_gradients(X: np.ndarray, y: np.ndarray, w: float, b: float) -> Tuple[float, float]:
    """Вычисляет градиенты по w и b"""
    n = len(y)
    predictions = w * X + b
    dw = (1 / n) * np.sum((predictions - y) * X)
    db = (1 / n) * np.sum(predictions - y)
    return dw, db


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 100,
    verbose: bool = True
) -> Tuple[float, float, list]:
    """Обычный градиентный спуск"""
    w = 0.0
    b = 0.0
    loss_history = []

    for i in range(n_iterations):
        loss = compute_loss(X, y, w, b)
        loss_history.append(loss)

        dw, db = compute_gradients(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db

        if verbose and (i % 10 == 0 or i == n_iterations - 1):
            print(f"Iteration {i:3d}: Loss = {loss:.2f}, w = {w:.3f}, b = {b:.3f}")

    return w, b, loss_history


def main():
    print("=" * 60)
    print("\nГенерируем данные y = 3x + 7 + шум...")

    X, y = generate_data(n_samples=1000, noise=10.0)
    print(f"Сгенерировано {len(X)} точек")

    print("\nЗапускаем градиентный спуск...")
    w, b, loss_history = gradient_descent(
        X, y,
        learning_rate=0.01,
        n_iterations=100,
        verbose=True
    )

    print("\n" + "=" * 60)
    print(f"Результат: w = {w:.3f}, b = {b:.3f}")
    print(f"Ожидалось: w ~= 3.000, b ~= 7.000")
    print("=" * 60)

    # Визуализация
    plt.figure(figsize=(12, 5))

    # График 1: Данные и предсказания
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, s=10, label='Данные')
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = w * x_line + b
    plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {w:.2f}x + {b:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Линейная регрессия')
    plt.grid(True, alpha=0.3)

    # График 2: История ошибки
    plt.subplot(1, 2, 2)
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Итерация')
    plt.ylabel('Loss (MSE)')
    plt.title('Сходимость градиентного спуска')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('gradient_descent_result.png', dpi=100)
    print("\nГрафик сохранён в gradient_descent_result.png")
    plt.show()


if __name__ == "__main__":
    main()
