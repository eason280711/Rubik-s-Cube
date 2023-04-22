
import argparse

# train parameters
batch_size, lr, beta1, beta2, epochs, total_epochs = 8, 0.00002, 0.5, 0.999, 1, 1

# model parameters
d_model, nhead, num_layers, dropout, seq_length, vocab_size = 256, 32, 7, 0.1, 54, 7

file = "train.py"

parser = argparse.ArgumentParser(prog="config.py", description="Config")
parser.add_argument("--run", type=str, default="train", help="file to run")
parser.add_argument("--batch_size", type=int, default=batch_size, help="batch size for training")
parser.add_argument("--lr", type=float, default=lr, help="learning rate")
parser.add_argument("--beta1", type=float, default=beta1, help="beta1 for Adam optimizer")
parser.add_argument("--beta2", type=float, default=beta2, help="beta2 for Adam optimizer")
parser.add_argument("--epochs", type=int, default=epochs, help="number of epochs to train")
parser.add_argument("--total_epochs", type=int, default=total_epochs, help="total number of epochs to train")

parser.add_argument("--d_model", type=int, default=d_model, help="dimension of model")
parser.add_argument("--nhead", type=int, default=nhead, help="number of heads in multi-head attention")
parser.add_argument("--num_layers", type=int, default=num_layers, help="number of layers in the transformer")
parser.add_argument("--dropout", type=float, default=dropout, help="dropout rate")
parser.add_argument("--seq_length", type=int, default=seq_length, help="sequence length")
parser.add_argument("--vocab_size", type=int, default=vocab_size, help="vocabulary size")

args = parser.parse_args()

file = args.run + ".py"

# train parameters
batch_size, lr, beta1, beta2, epochs, total_epochs = args.batch_size, args.lr, args.beta1, args.beta2, args.epochs, args.total_epochs

# model parameters
d_model, nhead, num_layers, dropout, seq_length, vocab_size = args.d_model, args.nhead, args.num_layers, args.dropout, args.seq_length, args.vocab_size

