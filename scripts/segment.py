import argparse
import logging

from pickybpe.segmentation import PickyBPECore


parser = argparse.ArgumentParser()
parser.add_argument('--bpe_model', type=str, required=True, help='Path to the BPE model.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input file.')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file.')
parser.add_argument('--return_type', type=str, default='str', help='Return type: str or int.')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model = PickyBPECore(args.bpe_model)
model.encode_file(args.input_file, args.output_file, args.return_type)
