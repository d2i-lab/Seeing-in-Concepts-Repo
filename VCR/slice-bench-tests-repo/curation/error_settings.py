from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import numpy.random as npr
import pandas as pd

def truncated_normal(mean, std_dev, size, min_val=0, max_val=1):
    samples = []
    while len(samples) < size:
        candidate_samples = np.random.normal(mean, std_dev, size)
        valid_samples = candidate_samples[(candidate_samples >= min_val) & (candidate_samples <= max_val)]
        samples.extend(valid_samples)
    samples = np.array(samples[:size])  # Trim to the desired size if excess samples were generated
    normalized_samples = (samples - min_val) / (max_val - min_val)
    return normalized_samples

class Distribution:
    def __call__(self, size: int):
        pass

@dataclass
class Normal(Distribution):
    loc: float
    scale: float
    def __call__(self, size: int, clip: bool = False):
        # result = npr.normal(self.loc, self.scale, size)
        result = truncated_normal(self.loc, self.scale, size)
        if clip:
            result = np.clip(result, 0, 1)
        return result

@dataclass
class NormalCenterRange(Distribution):
    scale: float
    center_range: Tuple[float, float]
    def __call__(self, size: int, clip: bool = False):
        # result = npr.normal(self.loc, self.scale, size)

        low, high = self.center_range
        center = np.random.randint(low*100, high*100, size) / 100
        result = truncated_normal(center, self.scale, size)
        if clip:
            result = np.clip(result, 0, 1)
        return result

@dataclass
class NormalGammaTail(Distribution):
    loc: float
    scale: float

    tail_percent: float
    gamma_shape: float
    gamma_scale: float

    def __call__(self, size: int, clip: bool = False):
        n_tail = int(self.tail_percent * size)
        n_normal = size - n_tail
        # result = np.random.normal(self.loc, self.scale, n_normal)
        result = truncated_normal(self.loc, self.scale, n_normal)
        tail = np.random.gamma(self.gamma_shape, self.gamma_scale, n_tail)
        result = np.concatenate((result, tail))
        if clip:
            result = np.clip(result, 0, 1)
        return result



@dataclass
class ErrorDescription:
    inlier: Optional[Distribution] = None
    outlier: Optional[Distribution] = None

# Mappings for loc and scale parameters for normal distribution
NORMAL_ERRORS_1 = {
    'low_separation': (
        ErrorDescription(
            inlier=Normal(loc=0.85, scale=0.2),
            outlier=Normal(loc=0.5, scale=0.2)
        )
    ),
    'medium_separation': (
        ErrorDescription(
            inlier=Normal(loc=0.85, scale=0.175),
            outlier=Normal(loc=0.45, scale=0.2)
        )
    ),
    'high_separation': (
        ErrorDescription(
            inlier=Normal(loc=0.85, scale=0.175),
            outlier=Normal(loc=0.3, scale=0.2)
        )
    ),
}

NORMAL_TAIL_ERRORS_1 = {
    'small_tail': (
        ErrorDescription(
            inlier=NormalGammaTail(
                loc=0.85, scale=0.2,
                tail_percent=0.1, gamma_shape=0.8, gamma_scale=0.5
            ),
            outlier=Normal(loc=0.25, scale=0.1)
        )
    ),
    'medium_tail': (
        ErrorDescription(
            inlier=NormalGammaTail(
                loc=0.85, scale=0.2,
                tail_percent=0.1, gamma_shape=0.8, gamma_scale=0.3
            ),
            outlier=Normal(loc=0.25, scale=0.1)
        )
    ),
    'high_tail': (
        ErrorDescription(
            inlier=NormalGammaTail(
                loc=0.85, scale=0.2,
                tail_percent=0.1, gamma_shape=0.8, gamma_scale=0.2
            ),
            outlier=Normal(loc=0.25, scale=0.1)
        )
    )
}

NORMAL_ERRORS_OUT_1 = {
    'low': (
        ErrorDescription(
            outlier=NormalCenterRange(center_range=(0.01, 0.25), scale=0.1)
        )
    ),
    'high': (
        ErrorDescription(
            outlier=NormalCenterRange(center_range=(0.251, 0.5), scale=0.1)
        )
    )
}


def get_error_types():
    return {
        'normal': NORMAL_ERRORS_1,
        'normal_tail': NORMAL_TAIL_ERRORS_1,
        'normal_outlier_only': NORMAL_ERRORS_OUT_1,
    }


def visualize_error(error_description: ErrorDescription, size: int, clip: bool=False):
    inlier = error_description.inlier(size, clip=clip)
    outlier = error_description.outlier(size, clip=clip)
    print(len(inlier), len(outlier))
    df = pd.DataFrame({'inlier': inlier, 'outlier': outlier})
    df.plot(kind='kde')
