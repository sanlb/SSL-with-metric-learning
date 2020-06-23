import torch
from params_utils.params import parse_dict_args
from runs import main


def parameters():
    defaults = {
        # Technical details
        'workers': 4,
        'checkpoint_epochs': 200,
        'method': 'mean_teacher',
        # Data
        'dataset': 'cifar10',
        'datadir': 'C:/data/',
        'base_batch_size': 100,
        # Architecture
        'arch': 'cnn13',
        'rampup_epoch': 80,
        'rampdown_epoch': 50,

        'base_lr': 0.003,
    }



    for data_seed in range(1000, 1010):
        yield {
            **defaults,
            'title': '4000-label cifar-10',
            'labels': 4000,
            'consistency': 100.0 * 4000. / 50000.,
            'data_seed': data_seed,
            'epochs': 300
        }




    # for data_seed in range(1008, 1010):
    #      yield {
    #          **defaults,
    #          'title': '4000-label cifar10',
    #          'n_labels': 4000,
    #          'consistency': 100.0 * 100. / 60000.,
    #          'data_seed': data_seed,
    #          'epochs': 300
    #      }

    #  # 2000 labels:
    # for data_seed in range(1000, 1010):
    #      yield {
    #          **defaults,
    #          'title': '2000-label cifar-10',
    #          'n_labels': 2000,
    #          'consistency': 100.0 * 2000. / 50000.,
    #          'data_seed': data_seed,
    #          'epochs': 300
    #      }


    # all labels:
    # for data_seed in range(1000, 1010):
    #    yield {
    #        **defaults,
    #        'title': 'all-label cifar-10',
    #        'n_labels': "all",
    #        'consistency': 100.0,
    #        'data_seed': data_seed,
    #        'epochs': 300
    #    }


def run(title, base_batch_size, base_lr, data_seed, **kwargs):
    print('run title: {title}, data seed: {seed}'.format(title=title, seed=data_seed))

    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."

    adapted_args = {
        'seed': data_seed,
        'batch_size': base_batch_size * ngpu,
        'lr': base_lr * ngpu,
    }

    main.main_args = parse_dict_args(**adapted_args, **kwargs)
    main.main()


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)