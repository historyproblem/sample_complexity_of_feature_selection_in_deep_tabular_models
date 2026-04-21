from ._channel_prob import ChannelZeroProbMetric

try:
    from ..models.feature_selection import get_stg_modules
except ImportError:
    get_stg_modules = None


class STGProbMetric(ChannelZeroProbMetric):
    def __init__(self, log_channel_zero_probs: bool = True):
        if get_stg_modules is None:
            raise RuntimeError(
                "STGProbMetric is unavailable in this build because STG selectors are not implemented."
            )
        super().__init__(
            get_stg_modules,
            log_channel_zero_probs=log_channel_zero_probs,
        )
