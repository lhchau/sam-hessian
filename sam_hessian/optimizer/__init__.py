from .sgd import SGD
from .sam import SAM
from .samdirection import SAMDIRECTION
from .sammagnitude import SAMMAGNITUDE
from .samanatomy import SAMANATOMY
from .usamanatomy import USAMANATOMY
from .usam import USAM
from .samckpt import SAMCKPT
from .samckptpos import SAMCKPTPOS
from .samckpt1 import SAMCKPT1
from .samckpt2 import SAMCKPT2
from .samckpt3 import SAMCKPT3
from .samckpt4 import SAMCKPT4
from .samckpt12 import SAMCKPT12
from .samckpt13 import SAMCKPT13
from .samckpt14 import SAMCKPT14
from .samckpt34 import SAMCKPT34
from .samckpt123 import SAMCKPT123
from .samckpt134 import SAMCKPT134
from .samckpt234 import SAMCKPT234
from .samexplore import SAMEXPLORE
from .sameckpt1 import SAMECKPT1
from .sameckpt2 import SAMECKPT2
from .sameckpt3 import SAMECKPT3
from .sameckpt4 import SAMECKPT4
from .sameckpt12 import SAMECKPT12
from .sameckpt13 import SAMECKPT13
from .sameckpt134 import SAMECKPT134
from .customsame import CUSTOMSAME
from .samckpt124 import SAMCKPT124
from .forgetsam import FORGETSAM
from .geosam import GEOSAM
from .clipsam import CLIPSAM
from .clipsamneg import CLIPSAMNEG
from .samslownoisy import SAMSLOWNOISY
from .sambelief import SAMBELIEF
from .samgc import SAMGC
from .customsame234mul import CUSTOMSAME234MUL
from .customsame2mul34mul import CUSTOMSAME2MUL34MUL


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
    elif opt_name == 'samckptpos':
        return SAMCKPTPOS(
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
    elif opt_name == 'samckpt14':
        return SAMCKPT14(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt34':
        return SAMCKPT34(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt123':
        return SAMCKPT123(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt124':
        return SAMCKPT124(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt134':
        return SAMCKPT134(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samckpt234':
        return SAMCKPT234(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samexplore':
        return SAMEXPLORE(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'sameckpt1':
        return SAMECKPT1(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'sameckpt2':
        return SAMECKPT2(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'sameckpt3':
        return SAMECKPT3(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'sameckpt4':
        return SAMECKPT4(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'sameckpt12':
        return SAMECKPT12(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'sameckpt13':
        return SAMECKPT13(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'sameckpt134':
        return SAMECKPT134(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'customsame':
        return CUSTOMSAME(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'forgetsam':
        return FORGETSAM(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'geosam':
        return GEOSAM(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'clipsam':
        return CLIPSAM(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'clipsamneg':
        return CLIPSAMNEG(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samslownoisy':
        return SAMSLOWNOISY(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'sambelief':
        return SAMBELIEF(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'samgc':
        return SAMGC(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'customsame234mul':
        return CUSTOMSAME234MUL(
            net.parameters(),
            **opt_hyperparameter
        )
    elif opt_name == 'customsame2mul34mul':
        return CUSTOMSAME2MUL34MUL(
            net.parameters(),
            **opt_hyperparameter
        )
    else:
        raise ValueError("Invalid optimizer!!!")