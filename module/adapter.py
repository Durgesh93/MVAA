"""
Per-voxel embedding adapter for the nnU-Net trunk.

Stock nnU-Net ends each decoder stage in a 1x1x1 conv ("seg_layer") that
maps stage features straight to num_segmentation_heads class logits. This
adapter splits that into two steps: the trunk is built to emit
`embedding_dim` channels instead of class logits, and a single shared
1x1x1 conv maps those to the class logits everything downstream expects.

The point is a named D-dimensional embedding per voxel that a
clustering-style loss can attach to -- prototypes, soft k-means, the CM
family -- in the same role `pre_heatmaps` plays in the supercm project,
while the segmentation path keeps exactly the shape and semantics it had.

What this does NOT do is add capacity. There is no nonlinearity between
the trunk's seg_layer and the classifier, so their composition is still a
linear map from stage features to logits, and with embedding_dim (128)
far above num_classes (3) it is a full-rank reparameterization: the
network can represent precisely what it could before. The adapter buys an
embedding space to put a loss on, not expressiveness. That also means
turning it on should not move the supervised baseline.

Deep supervision is passed through untouched: with it on, the trunk
returns one embedding tensor per decoder stage (highest resolution
first) and the classifier is applied to each, sharing weights across
scales. A clustering loss almost certainly wants embeddings[0] alone --
the other scales are coarser views of the same space, and mixing
resolutions into one set of prototypes would weight the deep, tiny maps
equally with the full-resolution one.
"""

from contextlib import contextmanager

from torch import nn


class EmbeddingAdapter(nn.Module):
    """
    nnU-Net trunk emitting embeddings, plus a linear head to class logits.

    forward(x) returns class logits, so this is a drop-in replacement for
    the bare network: the loss, the deep-supervision wrapper and
    nnUNetPredictor all see what they saw before. Set return_embeddings
    (or use the embeddings_enabled context manager) to get
    (embeddings, logits) instead.

    backbone: a network built with num_output_channels=embedding_dim.
    conv_op:  the conv class matching the task's dimensionality, e.g.
              nn.Conv3d -- read off the trunk rather than hardcoded, so a
              2D configuration works without a special case.
    """

    def __init__(self, backbone, embedding_dim, num_classes, conv_op):
        super().__init__()

        self.backbone = backbone
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)

        self.classifier = conv_op(self.embedding_dim, self.num_classes, kernel_size=1, bias=True)

        # Off by default so that every existing caller -- the predictor
        # included -- keeps getting plain logits. Flip it through
        # embeddings_enabled() rather than assigning directly.
        self.return_embeddings = False

    @property
    def decoder(self):
        """
        The trunk's decoder.

        PredictionOps._predict_logits toggles `net.decoder.deep_supervision`
        around sliding-window inference, and DDPHelper-unwrapped networks go
        through the same path, so the attribute has to survive the wrapping.
        Read-only: callers mutate the decoder, never replace it.
        """
        return self.backbone.decoder

    def _classify(self, embeddings):
        if isinstance(embeddings, (list, tuple)):
            return [self.classifier(e) for e in embeddings]

        return self.classifier(embeddings)

    def forward(self, x):
        embeddings = self.backbone(x)
        logits = self._classify(embeddings)

        if self.return_embeddings:
            return embeddings, logits

        return logits


@contextmanager
def embeddings_enabled(network):
    """
    Make network(x) return (embeddings, logits) for the duration.

    Inside the block, keep calling the network you were handed --
    network(x) -- and never the unwrapped `.module`. Under DDP only the
    wrapper's forward registers the gradient-sync hooks, so a forward that
    needs gradients and goes around it silently desyncs the ranks.

    Restores the previous flag on the way out, including on an exception,
    so a failed step cannot leave the predictor receiving tuples.
    """
    adapter = network.module if hasattr(network, "module") else network

    if not isinstance(adapter, EmbeddingAdapter):
        raise TypeError(
            f"embeddings_enabled() needs an EmbeddingAdapter, got {type(adapter).__name__}. "
            "NNUnetSetup.build_network() always returns one, so this means the network was "
            "replaced or built elsewhere."
        )

    previous = adapter.return_embeddings
    adapter.return_embeddings = True

    try:
        yield network
    finally:
        adapter.return_embeddings = previous
