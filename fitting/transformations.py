import copy

import fitting.regression as regression
import torch


class LinearTransform:
    def __init__(self, slope, intercept=None):
        self.slope = torch.atleast_1d(slope)
        if intercept is None:
            self.intercept = torch.zeros_like(self.slope)
        else:
            self.intercept = torch.atleast_1d(intercept)

    def transformData(self, *data):
        if len(data) == 1:
            data = data[0]
            return (data - self.intercept) / self.slope
        else:
            ret = tuple(
                (d - self.intercept[i]) / self.slope[i] for i, d in enumerate(data)
            )
            return ret

    def iTransformData(self, *data):
        if len(data) == 1:
            data = data[0]
            return (data * self.slope) + self.intercept
        else:
            return tuple(
                (d * self.slope[i]) + self.intercept[i] for i, d in enumerate(data)
            )

    def transformVariances(self, v):
        # return v / self.slope
        return v / self.slope**2

    def iTransformVariances(self, v):
        # return v * self.slope
        return v * self.slope**2

    def __repr__(self):
        return f"LinearTransform({self.slope}, {self.intercept})"

    def toCuda(self):
        return LinearTransform(self.slope.cuda(), self.intercept.cuda())

    def toNumpy(self):
        x = copy.deepcopy(self)
        x.slope = x.slope.cpu().numpy()
        x.intercept = x.intercept.cpu().numpy()
        return x


class AnscombeTransform:
    def __init__(self, slope, intercept=None):
        self.slope = torch.atleast_1d(slope)
        self.intercept = torch.atleast_1d(intercept)

    def transformData(self, y):
        return 2 * torch.sqrt(y + 3 / 8)

    def iTransformData(self, y):
        return (y / 2) ** 2 - 3 / 8

    def transformVariances(self, v):
        return v

    def iTransformVariances(self, v):
        return v


class DataTransformation:
    def __init__(self, transform_x, transform_y):
        self.transform_x = transform_x
        self.transform_y = transform_y

    def toCuda(self):
        return DataTransformation(self.transform_x.toCuda(), self.transform_y.toCuda())

    def transformX(self, edges, X):
        return (
            self.transform_x.transformData(*edges),
            self.transform_x.transformData(X),
        )

    def transformY(self, Y, V):
        return (
            self.transform_y.transformData(Y),
            self.transform_y.transformVariances(V),
        )

    def transform(self, dv):
        E, X = self.transformX(dv.E, dv.X)
        Y, V = self.transformY(dv.Y, dv.V)
        return regression.DataValues(X, Y, V, E)

    def iTransformX(self, edges, X):
        return (
            self.transform_x.iTransformData(*edges),
            self.transform_x.iTransformData(X),
        )

    def iTransformY(self, Y, V):
        return (
            self.transform_y.iTransformData(Y),
            self.transform_y.iTransformVariances(V),
        )

    def iTransform(self, dv):
        E, X = self.iTransformX(dv.E, dv.X)
        Y, V = self.iTransformY(dv.Y, dv.V)
        return regression.DataValues(X, Y, V, E)

    def __repr__(self):
        return f"DataTransformation({self.transform_x}, {self.transform_y})"


def getNormalizationTransform(dv, scale=1.0) -> DataTransformation:
    X, Y, V, E = dv.X, dv.Y, dv.V, dv.E

    max_x, min_x = torch.max(X, axis=0).values, torch.min(X, axis=0).values
    max_y, min_y = torch.max(Y), torch.min(Y)
    mean_y = torch.mean(Y)
    std_y = Y.std(dim=-1)

    # value_scale = max_y - min_y
    # print(f"MaxScale is : {value_scale}")
    value_scale = std_y
    # print(f"Std is: {std_y}")
    # input_scale = max_x - min_x

    transform_x = LinearTransform(scale * (max_x - min_x), min_x)
    transform_y = LinearTransform((max_y - min_y), min_y)
    transform_y = LinearTransform(std_y, mean_y)

    # transform_y = AnscombeTransform(torch.tensor([100.0]), torch.Tensor([0.0]))
    # transform_y = LinearTransform( std_y, mean_y)
    # transform_y = LinearTransform(max_y, torch.Tensor([0.0]))
    # transform_y = LinearTransform(torch.Tensor([1.0]), torch.Tensor([0.0]))
    # transform_y = LogTransform()

    # transform_y = LinearTransform(torch.tensor([1.0]), torch.tensor([1.0]))
    # transform_y = LinearTransform(scale * value_scale, min_y)

    return DataTransformation(transform_x, transform_y)
