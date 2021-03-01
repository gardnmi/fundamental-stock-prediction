from skorch import NeuralNetRegressor
from skorch.callbacks import EpochScoring


def r_squared(model, X, y_true, multioutput="uniform_average", sample_weight=None):
    y_pred = model.predict(X)[:, 1].reshape(-1, 1)
    weight = 1.

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (weight * (y_true - np.average(y_true, axis=0,
                                                 weights=sample_weight)) ** 2).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - \
        (numerator[valid_score] / denominator[valid_score])

    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0

    return np.average(output_scores, weights=avg_weights)


r2 = EpochScoring(scoring=r_squared, lower_is_better=False)


class QuantileRegressorModel(nn.Module):

    def __init__(self, n_cont, out_sz, layers, p=0.9):
        super().__init__()

        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_in = n_cont
        layerlist = []

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))

        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cont):
        x = self.bn_cont(x_cont)
        x = self.layers(x)
        return x


class QuantileRegressorNet(NeuralNetRegressor):
    def get_loss(self, preds, target, quantiles=(0.2, 0.5, 0.8), *args, **kwargs):
        #         assert not target.requires_grad
        #         assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


net = QuantileRegressorNet(
    module=QuantileRegressorModel,
    module__n_cont=X_train.shape[1],
    module__out_sz=3,
    module__layers=[100],
    lr=0.1,
    callbacks=[r2],
)

net.fit(X.to_numpy().astype(np.float32),
        y.to_numpy().reshape(-1, 1).astype(np.float32))
