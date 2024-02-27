from .utils.base import MLPBase


from .depth_route_net import DepthRouteNet
from .flatten_net import FlattenDepthRouteNet

NETWORK_DICT = {
    "DepthRouteNet": DepthRouteNet,
    "FlattenDepthRouteNet": FlattenDepthRouteNet,
}

BASENET_DICT = {
    "mlp": MLPBase,
}
