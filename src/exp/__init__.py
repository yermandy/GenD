from . import (
    examples,
    third_party,
    wacv_rebuttal,
    wacv_rebuttal_aug_robustness,
    wacv_rebuttal_paired_unpaired,
)

experiments = {
    **examples.experiments,
    **third_party.experiments,
    **wacv_rebuttal.experiments,
    **wacv_rebuttal_paired_unpaired.experiments,
    **wacv_rebuttal_aug_robustness.experiments,
}
