import sys


class Peel:
    """
    A 1D peel around a matrix
    """
    def __init__(self, inner, m, n):
        self.inner = inner
        self.m = m
        self.n = n

    @staticmethod
    def get_peel_path(m, n):
        """
        Generates a list of tuple indices for the outermost peel of a matrix of size nxm
        """
        il = ([0] * (n - 1)) + list(range(m)) + ([m - 1] * (n - 2)) + list(range(m - 1, 0, -1))
        jl = list(range(n)) + ([n - 1] * (m - 2)) + list(range(n - 1, -1, -1)) + ([0] * (m - 2))
        return list(zip(il, jl))

    def rotated(self, r):
        """
        Generates a new peel rotated r times (counter-clockwise)
        """
        return Peel(
            inner=[
                self.inner[(i + r) % len(self.inner)]
                for i, x in enumerate(self.inner)
            ],
            m=self.m,
            n=self.n,
        )

    def __repr__(self):
        return "Peel({})".format(self.inner)


class Matrix:
    """
    Representation of a 2D, mxn matrix
    """
    def __init__(self, inner):
        self.inner = inner

    @property
    def m(self):
        return len(self.inner)

    @property
    def n(self):
        return len(self.inner[0]) if len(self.inner) > 0 else 0

    def is_empty(self):
        return self.m == 0 or self.n == 0

    def peel(self):
        """
        Generates a list of peels for the matrix
        """

        def peel_once(matrix):
            peel = Peel(
                inner=[matrix.inner[i][j] for i, j in Peel.get_peel_path(matrix.m, matrix.n)],
                m=matrix.m,
                n=matrix.n,
            )
            core = [
                [matrix.inner[i][j] for j in range(1, matrix.n - 1)]
                for i in range(1, matrix.m - 1)
            ]
            return peel, Matrix(core)

        peels = []
        current_matrix = self
        while not current_matrix.is_empty():
            new_peel, new_matrix = peel_once(current_matrix)
            peels.append(new_peel)
            current_matrix = new_matrix

        return peels

    @staticmethod
    def from_peels(peels):
        """
        Creates a new matrix from reassembling peels
        """
        def from_peel_once(peel, matrix):
            new_matrix_inner = [
                [None for _ in range(peel.n)]
                for _ in range(peel.m)
            ]
            for peel_value, (i, j) in zip(peel.inner, Peel.get_peel_path(peel.m, peel.n)):
                new_matrix_inner[i][j] = peel_value

            for i in range(matrix.m):
                for j in range(matrix.n):
                    new_matrix_inner[i + 1][j + 1] = matrix.inner[i][j]
            return Matrix(new_matrix_inner)

        current_matrix = Matrix([])
        for current_peel in reversed(peels):
            current_matrix = from_peel_once(current_peel, current_matrix)

        return current_matrix

    def __repr__(self):
        return "Matrix({})".format(self.inner)

    def __str__(self):
        return "\n".join(
            " ".join(str(c) for c in r)
            for r in self.inner
        )


def parse_input():
    lines = list(sys.stdin)
    mnr = [int(t) for t in lines[0].strip().split()]
    r = mnr[2]

    return r, Matrix([
        [int(t) for t in line.strip().split()]
        for line in lines[1:]
    ])


def main():
    r, matrix = parse_input()
    peels = matrix.peel()
    rotated_peels = [p.rotated(r) for p in peels]
    new_matrix = Matrix.from_peels(rotated_peels)
    print(new_matrix)


if __name__ == "__main__":
    main()
