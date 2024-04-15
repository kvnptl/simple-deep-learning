
from torch import randint, tensor
import torchmetrics

target = randint(0, 2, (10, 25, 25))
pred = tensor(target)

pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]

jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=2)
ans = jaccard(pred, target)
print(ans)