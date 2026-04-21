from ..models.feature_selection import get_gumbel_modules
from ._channel_prob import ChannelZeroProbMetric


class GumbelProbMetric(ChannelZeroProbMetric):
    def __init__(self, log_channel_zero_probs: bool = True):
        super().__init__(
            get_gumbel_modules,
            log_channel_zero_probs=log_channel_zero_probs,
        )
