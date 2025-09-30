from pgmpy.sampling import BayesianModelSampling
from collections import Counter
from model import model

sampler = BayesianModelSampling(model)

print("=" * 60)
print("REJECTION SAMPLING: Appointment given Train=delayed")
print("=" * 60)