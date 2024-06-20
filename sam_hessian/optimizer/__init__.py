from .sgd import SGD
from .sam import SAM
from .samdirection import SAMDIRECTION
from .sammagnitude import SAMMAGNITUDE
from .sgdhess import SGDHESS
from .sgdvar import SGDVAR
from .ekfac import EKFAC
from .sgdsam import SGDSAM
from .samhess import SAMHESS
from .samanatomy import SAMANATOMY

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
    elif opt_name == 'sgdsam':
        return SGDSAM(
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
    elif opt_name == 'samhess':
        return SAMHESS(
            net.parameters(), 
            **opt_hyperparameter
        )
    elif opt_name == 'samanatomy':
        return SAMANATOMY(
            net.parameters(),
            **opt_hyperparameter
        )
    else:
        raise ValueError("Invalid optimizer!!!")