r"""Convert Torch models into export file formats like JSON.

This is a function used by Sven to extract the policy networks from trained
Actor-Critic modules and convert it to the JSON file format holding NN parameter
values.

Important Note:
    this file assumes that you are using the CrazyFlie Firmware adopted the the
    Chair of Data Processing - Technical University Munich (TUM)
"""

import argparse
import os
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.utils import export



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Name path of the file to be converted.}')
    parser.add_argument('--output', type=str, default='json',
                        help='Choose output file format: [onnx, json].}')
    args = parser.parse_args()

    assert os.path.exists(args.ckpt)
    if args.output == 'onnx':
        # load a saved checkpoint file (.pt), extract the actor network from
        # the ActorCritic module and save as .ONXX file to disk space
        export.convert_to_onxx_file_format(args.ckpt)
    elif args.output == 'json':
        # Convert PyTorch module to JSON file and save to disk.
        ac, env = utils.load_actor_critic_and_env_from_disk(args.ckpt)
        print(f'file_name_path=args.ckpt: {args.ckpt}')
        export.convert_actor_critic_to_json(
            actor_critic=ac,
            file_path=args.ckpt
        )

    else:
        raise ValueError('Expecting json or onnx as file output.')
