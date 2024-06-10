from .sgd import SGD
from .sam import SAM
from .samdirection import SAMDIRECTION
from .sammagnitude import SAMMAGNITUDE
from .sgdhess import SGDHESS
from .sgdvar import SGDVAR
from .ekfac import EKFAC

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
    elif opt_name == 'sgdhess':
        return SGDHESS(
            net.parameters(), 
            **opt_hyperparameter
        )
    elif opt_name == 'sgdvar':
        return SGDVAR(
            net.parameters(), 
            **opt_hyperparameter
        )
    elif opt_name == 'ekfac':
        return EKFAC(
            net, 
            **opt_hyperparameter
        )
    else:
        raise ValueError("Invalid optimizer!!!")