from jax import numpy as jnp


def humanlike_init(params):
    params_ = params.copy()
    params_["GDNGamma_0"]["bias"] = jnp.ones_like(params_["GDNGamma_0"]["bias"]) * 0.1
    params_["GDNGamma_0"]["kernel"] = (
        jnp.ones_like(params_["GDNGamma_0"]["kernel"]) * 0.5
    )

    ## Gabor
    params_["GaborLayerGammaHumanLike__0"]["freq_a"] = jnp.array([2.0, 4.0, 8.0, 16.0])

    params_["GaborLayerGammaHumanLike__0"]["gammax_a"] = (
        params_["GaborLayerGammaHumanLike__0"]["freq_a"] ** 0.9
    )
    params_["GaborLayerGammaHumanLike__0"]["gammay_a"] = (
        0.8 * params_["GaborLayerGammaHumanLike__0"]["gammax_a"]
    )

    ## GDNSpatioChromaFreqOrient
    params_["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"] = (
        jnp.ones_like(
            params_["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"]
        )
        * (1.0 / 0.2)
    )
    params_["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"][
        "gamma_theta_a"
    ] = jnp.ones_like(
        params_["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"][
            "gamma_theta_a"
        ]
    ) * (1 / (20 * jnp.pi / 180))
    return params_
