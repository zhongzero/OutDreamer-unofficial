
from .outdreamer.modeling_opensora import OpenSora_models
from .udit.modeling_udit import UDiT_models
from .outdreamer.modeling_cnext import OpenSoraCNext_models

Diffusion_models = {}
Diffusion_models.update(OpenSora_models)
Diffusion_models.update(UDiT_models)
Diffusion_models.update(OpenSoraCNext_models)

from .outdreamer.modeling_opensora import OpenSora_models_class
from .udit.modeling_udit import UDiT_models_class
from .outdreamer.modeling_cnext import OpenSoraCNext_models_class

Diffusion_models_class = {}
Diffusion_models_class.update(OpenSora_models_class)
Diffusion_models_class.update(UDiT_models_class)
Diffusion_models_class.update(OpenSoraCNext_models_class)