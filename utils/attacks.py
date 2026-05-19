from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    CarliniL2Method,
    DeepFool,
)


def get_attack(name, classifier, epsilon):

    if name == "FGSM":

        return FastGradientMethod(
            estimator=classifier,
            eps=epsilon,
        )

    elif name == "PGD":

        return ProjectedGradientDescent(
            estimator=classifier,
            eps=epsilon,
            eps_step=0.01,
            max_iter=40,
        )

    elif name == "C&W":

        return CarliniL2Method(
            classifier=classifier,
            max_iter=10,
            confidence=0.0,
        )

    elif name == "DeepFool":

        return DeepFool(
            classifier=classifier,
            max_iter=50,
        )
