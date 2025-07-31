import conv_nn as conv

from waggon import functions as f
from waggon.optim import SurrogateOptimiser
from waggon.surrogates import GP, DGP
from waggon.acquisitions import CB

import json

opt = SurrogateOptimiser(
    func=conv.ConvNN(),
    surr=GP(n_epochs=10),
    acqf=CB(),
    seed=2,
    max_iter=10,
    plot_results=True,
    verbose=2
)

result = opt.optimise()

best_result = {
    'x': result.x.tolist(),
    'fun': result.fun.item(),
    'success': result.success,
    'message': result.message
}

with open("optimisation_result.json", 'w') as f:
    json.dump(best_result, f, indent=2)

# print(f"Saved optimisation result to optimisation_result.json")

print(result)
