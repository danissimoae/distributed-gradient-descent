#!/usr/bin/env python3
"""
Coordinator - управляет распределённым обучением
1. Генерирует данные и режет на куски для воркеров
2. Рассылает текущие параметры модели (w, b)
3. Собирает градиенты от воркеров
4. Усредняет их и обновляет параметры
5. Отдаёт метрики для веб-интерфейса
"""

import numpy as np
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dataclasses import dataclass, asdict
from typing import List, Dict
import threading
import logging
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

app = FastAPI(title="Distributed Gradient Descent Coordinator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models для API
class WorkerRegistration(BaseModel):
    worker_id: str


class GradientSubmission(BaseModel):
    worker_id: str
    dw: float
    db: float
    n_samples: int
    compute_time: float


@dataclass
class TrainingState:
    """Состояние обучения"""
    iteration: int = 0
    w: float = 0.0
    b: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.01
    n_workers: int = 0
    workers_ready: int = 0


@dataclass
class GradientUpdate:
    """Градиент от воркера"""
    worker_id: str
    dw: float
    db: float
    n_samples: int
    compute_time: float


class Coordinator:
    def __init__(self, n_samples: int = 10000, learning_rate: float = 0.01):
        self.state = TrainingState(learning_rate=learning_rate)
        self.lock = threading.Lock()

        # Генерируем данные один раз
        np.random.seed(42)
        X = np.random.randn(n_samples, 1) * 10
        y = 3 * X + 7 + np.random.randn(n_samples, 1) * 10

        self.X_full = X
        self.y_full = y

        # История для графиков
        self.history = {
            'iterations': [],
            'loss': [],
            'w': [],
            'b': [],
            'timestamps': []
        }

        # Буфер для градиентов от воркеров
        self.gradient_buffer: List[GradientUpdate] = []
        self.registered_workers: set = set()

        logging.info(f"Coordinator initialized with {n_samples} samples")

    def register_worker(self, worker_id: str) -> Dict:
        """Регистрирует воркера и выдаёт ему кусок данных"""
        with self.lock:
            if worker_id in self.registered_workers:
                logging.warning(f"Worker {worker_id} already registered")
            else:
                self.registered_workers.add(worker_id)
                self.state.n_workers = len(self.registered_workers)
                logging.info(f"Worker {worker_id} registered. Total workers: {self.state.n_workers}")

        # Делим данные поровну между воркерами
        worker_idx = list(self.registered_workers).index(worker_id)
        chunk_size = len(self.X_full) // self.state.n_workers
        start_idx = worker_idx * chunk_size
        end_idx = start_idx + chunk_size if worker_idx < self.state.n_workers - 1 else len(self.X_full)

        X_chunk = self.X_full[start_idx:end_idx]
        y_chunk = self.y_full[start_idx:end_idx]

        return {
            'worker_id': worker_id,
            'X': X_chunk.tolist(),
            'y': y_chunk.tolist(),
            'chunk_size': len(X_chunk)
        }

    def get_current_params(self) -> Dict:
        """Отдаёт текущие параметры модели"""
        with self.lock:
            return {
                'iteration': self.state.iteration,
                'w': self.state.w,
                'b': self.state.b,
                'learning_rate': self.state.learning_rate
            }

    def submit_gradient(self, gradient: GradientUpdate):
        """Воркер отправляет свой градиент"""
        with self.lock:
            self.gradient_buffer.append(gradient)
            logging.info(f"Gradient from {gradient.worker_id}: dw={gradient.dw:.4f}, db={gradient.db:.4f}, time={gradient.compute_time:.3f}s")

            # Если все воркеры прислали градиенты - агрегируем
            if len(self.gradient_buffer) == self.state.n_workers:
                self._aggregate_and_update()
                self.gradient_buffer.clear()

    def _aggregate_and_update(self):
        """Усредняет градиенты и обновляет параметры (вызывается под lock)"""
        # Weighted average по количеству сэмплов
        total_samples = sum(g.n_samples for g in self.gradient_buffer)
        dw_avg = sum(g.dw * g.n_samples for g in self.gradient_buffer) / total_samples
        db_avg = sum(g.db * g.n_samples for g in self.gradient_buffer) / total_samples

        # Обновляем параметры
        self.state.w -= self.state.learning_rate * dw_avg
        self.state.b -= self.state.learning_rate * db_avg

        # Считаем loss на всех данных
        predictions = self.state.w * self.X_full + self.state.b
        self.state.loss = float(np.mean((predictions - self.y_full) ** 2) / 2)

        # Сохраняем в историю
        self.state.iteration += 1
        self.history['iterations'].append(self.state.iteration)
        self.history['loss'].append(self.state.loss)
        self.history['w'].append(self.state.w)
        self.history['b'].append(self.state.b)
        self.history['timestamps'].append(time.time())

        logging.info(f"Iteration {self.state.iteration}: loss={self.state.loss:.2f}, w={self.state.w:.3f}, b={self.state.b:.3f}")

    def get_history(self) -> Dict:
        """Возвращает историю для веб-интерфейса"""
        with self.lock:
            return {
                'history': self.history,
                'current_state': asdict(self.state)
            }


# Глобальный координатор
coordinator = Coordinator(n_samples=10000, learning_rate=0.01)


# API endpoints
@app.post("/api/register")
async def register_worker(registration: WorkerRegistration):
    """Воркер регистрируется и получает свой кусок данных"""
    chunk_info = coordinator.register_worker(registration.worker_id)
    return chunk_info


@app.get("/api/params")
async def get_params():
    """Воркер запрашивает текущие параметры"""
    return coordinator.get_current_params()


@app.post("/api/gradient")
async def submit_gradient(gradient: GradientSubmission):
    """Воркер отправляет вычисленный градиент"""
    gradient_update = GradientUpdate(
        worker_id=gradient.worker_id,
        dw=gradient.dw,
        db=gradient.db,
        n_samples=gradient.n_samples,
        compute_time=gradient.compute_time
    )
    coordinator.submit_gradient(gradient_update)
    return {'status': 'ok'}


@app.get("/api/history")
async def get_history():
    """Веб-интерфейс запрашивает историю обучения"""
    return coordinator.get_history()


@app.get("/health")
async def health():
    return {'status': 'ok', 'workers': coordinator.state.n_workers}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level='info')
