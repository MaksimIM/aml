from argparse import ArgumentParser


def get_all_args():
    parser = ArgumentParser(description="AML")
    parser.add_argument('--save_path_prefix', type=str,
                        default='/tmp/tmp_aml',
                        help='Path for logs and output')
    parser.add_argument('--load_checkpt', type=str, default=None,
                        help='Directory with checkpoint files')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device, e.g. "cpu" or "cuda:0"')
    parser.add_argument('--log_interval', type=int, default=500,
                        help='Interval for log messages')
    parser.add_argument('--visualization_interval', type=int, default=None,  # 5000
                        help='Interval for visualization')
    parser.add_argument('--visualization_points', type=int, default=2500,  # or 10K
                        help='Max num points to visualize')
    parser.add_argument('--maximal_visualization_iterations', type=int, default=20,  # or 10
                        help='Max iter to collect viz points')
    parser.add_argument('--visualization_type', type=str, default=None,
                        choices=['2D scatter',
                                 '3D scatter',
                                 '2D forward'],
                        help='Method of visualization')
    parser.add_argument('--visualization_data_indexes', nargs='+', type=int,
                        help='Which data indexes should be visualized')
    parser.add_argument('--data_source', type=str,
                        default='Ellipse',
                        choices=['Ellipse',
                                 'ConservativeBlock1D',
                                 'ConservativeBlock2D',
                                 'ConservativeBlockConstVel1D',
                                 'ConservativeBlockConstVel2D',
                                 'ConservativeBlock45Incline1D',
                                 'Block36InclineNoDrag1D',
                                 'Block1D',
                                 'Block2D',
                                 'ConservativeBlockOnIncline',
                                 'BlockOnIncline',
                                 'BlockOnInclineODE',
                                 ],
                        help='Choice of the data generating class')
    parser.add_argument('--noise_scale', type=float, default=0.0,
                        help='Scaling factor for random observation noise')
    parser.add_argument('--g_batch_size', type=int, default=1024,
                        help='Batch size for training g relations')
    parser.add_argument('--g_hidden_size', type=int, default=4,
                        help='Hidden layers size for g')
    parser.add_argument('--g_num_layers', type=int, default=3,
                        help='Number of hidden layers for g')
    parser.add_argument('--g_max_train_epochs', type=int, default=50000,
                        help='Max num training epochs for g')
    parser.add_argument('--g_pretrain_epochs', type=int, default=0,
                        help='Num pretraininig epochs for g')
    parser.add_argument('--g_lr', type=float, default=1e-4,
                        help='Learning rate for g relations')
    parser.add_argument('--max_relations', type=int, default=10,
                        help='Maximum number of relations to learn (g1,...,gk)')
    parser.add_argument('--transverse_beta', type=float, default=1,
                        help='Beta weight for transverse loss (None for auto)')
    parser.add_argument('--max_syzygies', type=int, default=0,
                        help='Maximum number of syzygies (f1,...,fn)')
    parser.add_argument('--syz_hidden_size', type=int, default=32,
                        help='Hidden layers size for syzygies')
    parser.add_argument('--syz_max_train_epochs', type=int, default=None,
                        help='Max train epochs for syzygies (None for auto)')
    parser.add_argument('--syz_move_g_beta', type=float, default=5.0,
                        help='How strong to move g relations away')
    parser.add_argument('--syz_beta', type=float, default=1.0,
                        help='Multiplier for syz loss')
    parser.add_argument('--ablation', action='store_true',
                        help='Use only on-manifold and gradient loss')
    args = parser.parse_args()
    return args
