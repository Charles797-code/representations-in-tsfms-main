from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Solar,
    Dataset_PEMS,
    Dataset_Weather,
)
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'Weather': Dataset_Weather,
}


def data_provider(args, flag):
    Data = data_dict[args.get('data', 'ETTh1')]
    timeenc = 0 if args.get('embed', 'timeF') != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.get('batch_size', 128)
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.get('batch_size', 128)
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.get('batch_size', 128)

    freq = args.get('freq', 'h')

    data_set = Data(
        root_path=args['root_path'],
        data_path=args['data_path'],
        flag=flag,
        size=[args['seq_len'], args.get('label_len', 0), args['pred_len']],
        features=args.get('features', 'S'),
        target=args.get('target', 'OT'),
        scale=True,
        timeenc=timeenc,
        freq=freq,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.get('num_workers', 0),
        drop_last=drop_last,
    )

    return data_set, data_loader
