# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

from uuid import UUID

from src.models.deeplearning.base import ModelEvaluator

CONV_UUID = UUID("f58f5f7c-9e44-11ec-90a1-34f64b38befe")
LSTM_UUID = UUID("4eadb85a9f0911ec8c9a34f64b38befe")

if __name__ == "__main__":
    evaluator, train_info = ModelEvaluator.restore_from_ckpt(LSTM_UUID)
    evaluator.validate(train_info, keep_pickles=False)
