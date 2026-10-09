"""Prespecified paired descriptive contrasts; never promotes a checkpoint."""
import math


def compare(reference, candidate):
    expected=set(range(20))
    if set(reference) != expected or set(candidate) != expected:
        raise ValueError('exactly20 matched init IDs required')
    if any(type(v) is not bool for v in list(reference.values())+list(candidate.values())):
        raise ValueError('outcomes must be boolean')
    gains=[i for i in range(20) if candidate[i] and not reference[i]]
    losses=[i for i in range(20) if reference[i] and not candidate[i]]
    discordant=len(gains)+len(losses)
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(len(gains),len(losses))+1))/2**discordant) if discordant else 1.
    return dict(reference_successes=sum(reference.values()), candidate_successes=sum(candidate.values()),
                gains=gains,losses=losses,exact_mcnemar_two_sided_p=p,
                automatic_checkpoint_selection=False,progression_gate_replaced=False)
