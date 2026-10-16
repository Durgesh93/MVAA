"""
Per-stage feature probe for a STOCK nnU-Net trunk.

An earlier version of this file wrapped the trunk in an EmbeddingAdapter: the
backbone was built to emit 128 channels instead of class logits, and a shared
1x1 classifier mapped those to logits. That is gone. It changed the
architecture, renamed every parameter under "backbone." plus "classifier.",
and inserted a second linear map with no nonlinearity between the two -- a
redundant factorisation whose only effect was to add reparameterisation
freedom the clustering loss then had to be defended against.

This probes instead. nnU-Net's decoder already ends each stage in a 1x1x1
"seg_layer" that maps that stage's features to class logits, so the features
feeding it ARE the per-voxel embedding. A forward hook on each seg_layer
captures its input without touching the architecture: the network stays
stock, parameter names stay stock, and stock checkpoints load.

Native widths, highest resolution first: 32, 64, 128, 256, 320. Note the
decoder is WIDEST at the coarsest stage, so the full-resolution embedding is
the narrowest one.
"""

from contextlib import contextmanager


def unwrap(network):
    """The module itself, through a DDP wrapper if there is one."""
    return network.module if hasattr(network, "module") else network


def stage_dims(network):
    """Feature width feeding each seg_layer, highest resolution first."""
    return [layer.in_channels for layer in unwrap(network).decoder.seg_layers[::-1]]


@contextmanager
def stage_features(network):
    """
    Collect each seg_layer's INPUT for the duration, highest resolution first.

    Yields a list that fills during the forward, so read it AFTER calling the
    network:

        with stage_features(net) as features:
            logits = net(x)
        embeddings = order_stages(features, logits)

    Call the network you were handed, never the unwrapped module -- under DDP
    only the wrapper's forward registers the gradient-sync hooks. The hooks
    here are attached to inner modules and fire either way.

    Hooks are removed on the way out, including on an exception, so a failed
    step cannot leave them accumulating across iterations.
    """

    captured = []
    handles = [
        layer.register_forward_hook(lambda module, inputs, output: captured.append(inputs[0]))
        for layer in unwrap(network).decoder.seg_layers
    ]

    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def order_stages(captured, logits):
    """
    Match the capture order to the order the decoder returns its outputs in.

    The decoder walks stages from the bottleneck up, so hooks fire coarsest
    first, and it then reverses seg_outputs before returning. With deep
    supervision off it runs only the last seg_layer, so there is one capture
    and `logits` is a bare tensor rather than a list.
    """

    if not isinstance(logits, (list, tuple)):
        return captured[0]

    return captured[::-1]
