from .sgd import SGD
from .sam import SAM
from .samdirection import SAMDIRECTION
from .sammagnitude import SAMMAGNITUDE
from .samanatomy import SAMANATOMY
from .usamanatomy import USAMANATOMY
from .usam import USAM
from .samckpt import SAMCKPT
from .samckpt1 import SAMCKPT1
from .samckpt2 import SAMCKPT2
from .samckpt3 import SAMCKPT3
from .samckpt4 import SAMCKPT4
from .samckpt12 import SAMCKPT12
from .samckpt13 import SAMCKPT13
from .samckpt134 import SAMCKPT134
from .shame import SHAME
from .same import SAME

def get_optimizer(
    net,
    opt_name='sam',
    opt_hyperparameter={}):
    if opt_name == 'sam':
        return SAM(
            net.parameters(), 
            **opt_hyperparameter
        )
    elif opt_name == 'sgd':
        return SGD(
            net.parameters(), 
            **opt_hyperparameter
        )
    elif opt_name == 'samdirection':
        return SAMDIRECTION(
            net.parameters(), 
            **opt_hyperparameter
        )
    elif opt_name == 'sammagnitude':
        return SAMMAGNITUDE(
            net.parameters(), 
            **opt_hyperparameter
        )
    elif opt_name == 'samanatomy':
        return SAMANATOMY(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'usamanatomy':
        return USAMANATOMY(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'usam':
        return USAM(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt':
        return SAMCKPT(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt1':
        return SAMCKPT1(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt2':
        return SAMCKPT2(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt3':
        return SAMCKPT3(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt4':
        return SAMCKPT4(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt12':
        return SAMCKPT12(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt13':
        return SAMCKPT13(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt134':
        return SAMCKPT134(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'shame':
        return SHAME(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'same':
        return SAME(
            net.parameters(),
            **opt_hyperparameter
        )
    else:
        raise ValueError("Invalid optimizer!!!")