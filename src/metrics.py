import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Iterator

# =========================
# dataclasses
# =========================

@dataclass
class WeightMetrics:
    shape: Tuple[int, ...]
    num_params: int
    l1_norm: float
    l2_norm: float
    fro_norm: float
    spectral_norm: Optional[float]
    stable_rank: Optional[float]
    condition_number: Optional[float]
    effective_rank: Optional[float]
    weight_sparsity: float


@dataclass
class ActivationMetrics:
    shape: Tuple[int, ...]
    mean: float
    std: float
    abs_mean: float
    sparsity: float
    dead_fraction: Optional[float]
    effective_rank: Optional[float]


@dataclass
class GradientMetrics:
    grad_l2_norm: float
    grad_abs_mean: float
    grad_sparsity: float


class WCollector:
    '''Собирает метрики весов слоя.'''

    def can_handle(self, m: nn.Module) -> bool:
        '''Проверяет, есть ли у слоя веса.'''
        return hasattr(m, 'weight') and m.weight is not None

    def collect(self, name: str, m: nn.Module):
        '''Считает метрики весов для одного слоя.'''
        w = m.weight.detach().float()

        shape = tuple(w.shape)
        num_params = w.numel()

        l1_norm = w.abs().sum().item()
        l2_norm = torch.linalg.vector_norm(w.reshape(-1), ord=2).item()
        fro_norm = torch.sqrt((w ** 2).sum()).item()
        # weight_sparsity = (w == 0).float().mean().item()
        eps = 1e-4
        weight_sparsity = (w.abs() < eps).float().mean().item()

        return WeightMetrics(
            shape=shape,
            num_params=num_params,
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            fro_norm=fro_norm,
            spectral_norm=None,
            stable_rank=None,
            condition_number=None,
            effective_rank=None,
            weight_sparsity=weight_sparsity,
        )


class ModelInspector:
    '''Обходит модель и запускает сбор метрик.'''

    def __init__(self, model: nn.Module):
        '''Сохраняет модель и инициализирует сборщики.'''
        self.model = model
        self.wc = WCollector()

    def iter_mods(self) -> Iterator[Tuple[str, nn.Module]]:
        '''Итерируется по всем модулям модели.'''
        for name, m in self.model.named_modules():
            yield name, m

    def total_params(self) -> int:
        '''Считает общее число параметров модели.'''
        return sum(p.numel() for p in self.model.parameters())
    
    def trainable_params(self) -> int:
        '''Считает число обучаемых параметров модели.'''
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _print_w_summary(self, res: Dict[str, 'WeightMetrics']) -> None:
        '''Печатает краткую сводку по weight-метрикам.'''
        print('=== Weight summary ===')
        print(f'layers with weights: {len(res)}')
        print(f'total params: {self.total_params()}')
        print(f'trainable params: {self.trainable_params()}')

        for name, wm in res.items():
            print(f'{name}: shape={wm.shape}, num_params={wm.num_params}')

    def collect_w(self, summary: bool = False) -> Dict[str, 'WeightMetrics']:
        '''Собирает метрики весов по всем подходящим слоям.'''
        res: Dict[str, 'WeightMetrics'] = {}

        for name, m in self.iter_mods():
            if self.wc.can_handle(m):
                wm = self.wc.collect(name, m)
                if wm is not None:
                    res[name] = wm
        
        if summary:
            self._print_w_summary(res)

        return res