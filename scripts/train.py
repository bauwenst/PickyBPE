import argparse
import logging

from pickybpe.vocabularisation import PickyBPETrainer


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, help='Path to the input file.')
parser.add_argument('--model_file', type=str, help='Path to the output model file.')
parser.add_argument('--vocab_size', type=int, help='Desired vocabulary size.')
parser.add_argument('--threshold', type=float, help='Desired threshold.')
parser.add_argument('--coverage', type=float, default=0.9999, help='Desired coverage.')
parser.add_argument('--pad_id', type=int, default=0, help='ID of the padding token.')
parser.add_argument('--unk_id', type=int, default=1, help='ID of the unknown token.')
parser.add_argument('--bos_id', type=int, default=2, help='ID of the beginning-of-sequence token.')
parser.add_argument('--eos_id', type=int, default=3, help='ID of the end-of-sequence token.')
parser.add_argument('--logging_step', type=int, default=200, help='Logging step.')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
trainer = PickyBPETrainer(
    args.vocab_size,
    pad_id=args.pad_id,
    unk_id=args.unk_id,
    bos_id=args.bos_id,
    eos_id=args.eos_id,
    coverage=args.coverage,
    threshold=args.threshold
)
trainer.fit(args.input_file, args.model_file, args.logging_step)
