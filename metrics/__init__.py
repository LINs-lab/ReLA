from metrics.lineareval import linear_eval
from metrics.segment import segment_eval

EVALUATORS = {
    "linear": linear_eval,
    "segment": segment_eval,
}
