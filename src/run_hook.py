import logging

from ml_serving.drivers import driver
import numpy as np

import hook


LOG = logging.getLogger(__name__)


def main():
    drv = driver.load_driver('null')
    serving = drv(
        preprocess=hook.preprocess,
        postprocess=hook.postprocess,
        init_hook=hook.init_hook,
        init_param='some-value',
    )
    serving.load_model('some_path')

    result = serving.predict_hooks({'input': np.random.rand(1, 28, 28, 1)})
    print('result shape: {}'.format(result['input'].shape))


if __name__ == '__main__':
    main()
