import warnings
from tune3.curvature.hutchinson import HutchinsonEstimator, HutchinsonConfig


def hutchinson_trace_estimator(model, loss, num_vectors=10):
    """
    DEPRECIADO. Use tune3.curvature.HutchinsonEstimator().estimate().

    ATENÇÃO: a versão anterior deste módulo calculava Tr(H) incorretamente
    via v^T H v. Esta versão delega para o estimador correto de Tr(H^2)
    via ||Hv||^2.
    """
    warnings.warn(
        "hutchinson_trace_estimator() está depreciado. "
        "Use HutchinsonEstimator de tune3.curvature. "
        "A versão anterior calculava Tr(H) — quantidade errada.",
        DeprecationWarning,
        stacklevel=2,
    )
    return HutchinsonEstimator(HutchinsonConfig(num_probes=num_vectors)).estimate(model, loss)