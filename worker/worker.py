#!/usr/bin/env python3
"""
Worker - вычисляет градиенты на своём куске данных
1. Регистрируется у координатора и получает данные
2. В цикле: берёт параметры, считает градиент, отправляет обратно
3. Экспортирует метрики времени вычисления
"""

import numpy as np
import httpx
import time
import sys
import logging
from typing import Tuple
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s [Worker] %(message)s')


class Worker:
    def __init__(self, worker_id: str, coordinator_url: str = "http://localhost:5000"):
        self.worker_id = worker_id
        self.coordinator_url = coordinator_url
        self.X = None
        self.y = None

        logging.info(f"[{worker_id}] Starting...")

    async def register(self):
        """Регистрируется у координатора и получает данные"""
        logging.info(f"[{self.worker_id}] Registering with coordinator...")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.coordinator_url}/api/register",
                json={'worker_id': self.worker_id},
                timeout=10.0
            )

            if response.status_code != 200:
                raise RuntimeError(f"Registration failed: {response.text}")

            data = response.json()
            self.X = np.array(data['X'])
            self.y = np.array(data['y'])

            logging.info(f"[{self.worker_id}] Registered. Got {len(self.X)} samples")

    def compute_gradient(self, w: float, b: float) -> Tuple[float, float]:
        """Вычисляет градиент на своих данных"""
        n = len(self.y)
        predictions = w * self.X + b
        dw = (1 / n) * np.sum((predictions - self.y) * self.X)
        db = (1 / n) * np.sum(predictions - self.y)
        return float(dw), float(db)

    async def run(self, n_iterations: int = 100):
        """Основной цикл воркера"""
        async with httpx.AsyncClient() as client:
            for iteration in range(n_iterations):
                # Получаем текущие параметры от координатора
                response = await client.get(
                    f"{self.coordinator_url}/api/params",
                    timeout=10.0
                )
                params = response.json()

                w = params['w']
                b = params['b']

                # Вычисляем градиент на своих данных
                start_time = time.time()
                dw, db = self.compute_gradient(w, b)
                compute_time = time.time() - start_time

                # Отправляем градиент координатору
                await client.post(
                    f"{self.coordinator_url}/api/gradient",
                    json={
                        'worker_id': self.worker_id,
                        'dw': dw,
                        'db': db,
                        'n_samples': len(self.X),
                        'compute_time': compute_time
                    },
                    timeout=10.0
                )

                logging.info(f"[{self.worker_id}] Iteration {iteration}: dw={dw:.4f}, db={db:.4f}, time={compute_time:.3f}s")

                # Небольшая задержка, чтобы координатор успел обновиться
                await asyncio.sleep(0.1)

        logging.info(f"[{self.worker_id}] Completed {n_iterations} iterations")


async def wait_for_coordinator(coordinator_url: str, worker_id: str, max_retries: int = 10):
    """Ждёт пока координатор запустится"""
    async with httpx.AsyncClient() as client:
        for i in range(max_retries):
            try:
                response = await client.get(f"{coordinator_url}/health", timeout=2.0)
                if response.status_code == 200:
                    return
            except:
                if i < max_retries - 1:
                    logging.info(f"[{worker_id}] Waiting for coordinator... ({i+1}/{max_retries})")
                    await asyncio.sleep(2)
                else:
                    logging.error(f"[{worker_id}] Coordinator not available")
                    sys.exit(1)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python worker.py <worker_id> [coordinator_url]")
        sys.exit(1)

    worker_id = sys.argv[1]
    coordinator_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5000"

    await wait_for_coordinator(coordinator_url, worker_id)

    worker = Worker(worker_id, coordinator_url)
    await worker.register()
    await worker.run(n_iterations=100)


if __name__ == '__main__':
    asyncio.run(main())
