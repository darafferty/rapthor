import numpy as np
from numpy.polynomial.polynomial import polyval as np_polyval


class LagrangePolynomial:
    def __init__(self, order=None, nr_coeffs=None):
        if (order is None and nr_coeffs is None) or (
            order is not None and nr_coeffs is not None
        ):
            raise ValueError("Initializer accepts only one input argument")
        elif order is not None:
            self._order = order
            self._nr_coeffs = self.compute_nr_coeffs(order)
        else:
            self._order = self.compute_order(nr_coeffs)
            self._nr_coeffs = nr_coeffs

    def evaluate(self, x, y, coeffs):
        """
        Evaluate the polynomial on a given set of coordinates, for a
        given set of coefficients.

        Parameters
        ----------
        x : [type]
            [description]
        y : [type]
            [description]
        coeffs : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        # Evaluate polynomial using Horner's method
        assert coeffs.size == self.nr_coeffs

        x = np.asarray(x) if not isinstance(x, np.ndarray) else x
        y = np.asarray(y) if not isinstance(y, np.ndarray) else y

        assert x.ndim <= 1 and y.ndim <= 1
        y_coeff = (
            np.empty(self._order + 1)
            if y.ndim == 0
            else np.empty((self._order + 1, y.size))
        )
        for i in range(self._order + 1):
            idcs = self.get_indices_right_diagonal(self._order, i)
            y_coeff[i] = np_polyval(y, coeffs[idcs])
        return np_polyval(x, y_coeff)

    def expand_basis(self, x, y):
        """
        Expand the basis on the give coordinates

        Parameters
        ----------
        x : scalar, np.ndarray
            [description]
        y : scalar, np.ndarray
            [description]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        TypeError
            [description]
        """

        if not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            # Scalar case
            X = np.asarray(x)
            Y = np.asarray(y)
            basis_functions = np.empty((self.nr_coeffs, 1))
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            assert x.shape == y.shape and x.ndim == 1
            X, Y = np.meshgrid(x, y)
            basis_functions = np.empty((self.nr_coeffs,) + X.shape)
        else:
            raise TypeError(
                "Method currently only supported for scalar coordinate or vector input"
            )

        # This nested loop can probably be implemented more efficiently, if needed.
        # See for example the PolynomialFeatures.fit_transform method in scikit-learn"
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        for n in range(self.order + 1):
            # Loop over polynomial degree (rows in Pascal's triangle)
            for k in range(n + 1):
                # Loop over unique entries per polynomial degree (columns in Pascal's triangle)
                offset = np.sum(np.arange(1, n + 1, 1)) + k
                basis_functions[offset, ...] = X ** (n - k) * Y ** k
        return basis_functions

    @property
    def nr_coeffs(self):
        return self._nr_coeffs

    @property
    def order(self):
        return self._order

    @staticmethod
    def compute_nr_coeffs(order):
        return (order + 1) * (order + 2) // 2

    @staticmethod
    def compute_order(nr_coeffs):
        # Solution to the quadratic expression (order + 1)(order + 2) / 2 = nr_coeffs,
        # for the positive discriminant
        return int((-3 + np.sqrt(1 + 8 * nr_coeffs)) // 2)

    @staticmethod
    def get_indices_right_diagonal(order, diag_nr):
        if diag_nr > order:
            raise ValueError("Diagonal number should be smaller than polynomial order")

        indices = [
            diag_nr
            if diag_nr == 0
            else LagrangePolynomial.compute_nr_coeffs(diag_nr - 1)
        ]
        for i in range(diag_nr + 1, order + 1):
            indices.append(indices[-1] + i + 1)
        return np.array(indices)


# TODO: make unit tests
def main():
    poly = LagrangePolynomial(order=1)
    # result = poly.evaluate(1, 2, coeffs=np.array([1, 2, 3]))
    x = 1
    y = np.array([1, 2])

    # X, Y = np.meshgrid(x, y)
    # print(X)
    # print(Y)
    result = poly.evaluate(x, y, coeffs=np.array([1, 2, 3]))
    # result = poly.evaluate(2, 3, coeffs=np.array([1, 2, 3, 4, 5, 6]))
    print(result)
    # print(np_polyval())


if __name__ == "__main__":
    main()
