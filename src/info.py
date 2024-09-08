from torchinfo import summary
import argparse

from modules.model import SteerNetWrapped, Seq2SeqWrapped, PilotNetWrapped

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        help="name of the model (pilotnet, seq2seq or steer)",
                        required=True)
    args = parser.parse_args()

    if args.model == "steer":
        print(summary(SteerNetWrapped("cuda")))
    elif args.model == "pilotnet":
        print(summary(PilotNetWrapped("cuda")))
    elif args.model == "seq2seq":
        print(summary(Seq2SeqWrapped("cuda")))

# steer: 6.3M
# seq2seq 5.9M (about ~5M is from RegNet)
# pilotnet: 0.8M
