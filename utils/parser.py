import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SHDF")

    parser.add_argument("--dataset", nargs="?", default="last-fm", help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")

    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')


    parser.add_argument("--cuda", type=int, default=1, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=1, help="gpu id")

    parser.add_argument("--inverse_r", type=bool, default=False, help="consider inverse relation or not")

    #parser.add_argument("--mode", nargs="?", default="complete", help="Choose a mode:[denoise,complete,mix]")
    parser.add_argument('--denoise_keep_rate', type=float, default=0.8, help='keep rate of the denoiseing procedure')
    parser.add_argument('--complete_rate', type=float, default=0.2, help='keep rate of the denoiseing procedure')


    # ================= for relation predictor ==================
    parser.add_argument('--rp_epochs', type=int, default=50, help='number of epochs for relation predictor')
    parser.add_argument('--embed_dim', type=int, default=64, help='embedding dim for relation predictor')

    # ================= for recommender ==================
    parser.add_argument('--pretrain_epochs', type=int, default=500, help='pretrain epochs')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='learning rate for pretraining')
    parser.add_argument('--rec_epochs', type=int, default=100, help='number of epochs for recommender')
    parser.add_argument('--rec_batch_size', type=int, default=1024, help='batch size for training recommender')
    parser.add_argument('--latdim', type=int, default=64, help='embedding dim for recommender')
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument("--node_dropout", type=int, default=1, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=int, default=1, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    return parser.parse_args()