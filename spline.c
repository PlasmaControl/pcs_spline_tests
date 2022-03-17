#include <stdbool.h>
#include <stddef.h>
#include "spline.h"
#include <stdio.h>

int spline_fit_natural(struct spline * spl, spline_t * scratch) {
	if (scratch == NULL)
		return 3;

	size_t const n = spl->n;
	spline_t * subdiag = scratch;
	spline_t * diag = scratch + n;
	spline_t * superdiag = diag + n;

	// 1st and nth equations from boundary conditions
	diag[0] = 1.0;
	superdiag[0] = 0.0;
	diag[n-1] = 1.0;
	subdiag[n-2] = -1.0;

	// diagonals
	for (size_t i = 1; i < n-1; ++i) {
		diag[i] = 2.0 * (spl->x[i+1] - spl->x[i-1]);
		superdiag[i]= spl->x[i+1] - spl->x[i];
		subdiag[i-1] = spl->x[i] - spl->x[i-1];
	}

	spl->c[0] = 0;
	spl->c[n-1] = 0;
	for (size_t i = 1; i < n-1; ++i)
		spl->c[i] = 3.0 * (
			((spl->y[i+1] - spl->y[i])   / (spl->x[i+1] - spl->x[i])) -
			((spl->y[i]   - spl->y[i-1]) / (spl->x[i]   - spl->x[i-1])));

	/*
	 * O(n) Tridiagonal system solver: Thomas algorithm.
	 * Note: not guaranteed to be stable and destroys original input. 
	 * Reference: http://www.industrial-maths.com/ms6021_thomas.pdf
	 * x -- input vector, function returns solution
	 * n -- number of equations
	 * a -- subdiagonal
	 * b -- main diagonal
	 * c -- superdiagonal
	 */
	void trilus(size_t n, spline_t x[n], spline_t a[n], spline_t b[n], spline_t c[n]) {
		// Forward sweep
		for (size_t i = 1; i < n; ++i) {
			spline_t const m = a[i-1] / b[i-1];
			b[i] = b[i] - (m * c[i-1]);
			x[i] = x[i] - (m * x[i-1]);
		}

		x[n-1] /= b[n-1];

		// Backwards sweep
		for(size_t i = n-1; i > 0; --i)
			x[i-1] = (x[i-1] - c[i-1] * x[i]) / b[i-1];
	}

	trilus(spl->n, spl->c, subdiag, diag, superdiag);

	return 0;
}

int spline_eval(size_t N, spline_t values[N], spline_t const eval_pts[N], struct spline spl, spline_t scratch[3 * spl.n]) {
	if (values == NULL)
		return 1;
	else if (eval_pts == NULL)
		return 2;

	if (scratch)
		spline_fit_natural(&spl, scratch);

	// Binary search for index
	size_t const M = spl.n;
	spline_t const * X = spl.x;
	spline_t const * Y = spl.y;
	spline_t const * C = spl.c;
	for (size_t i = 0; i < N; ++i) {
		size_t idx = 0;
		size_t high = M - 1;
		size_t low = 0;
		size_t mid = low + ((high - low) / 2);
		if (eval_pts[i] <= X[0])
			idx = 0;
		else if (eval_pts[i] >= X[M-1])
			idx = M - 2;
		else
			while (low < high) {
				mid = low + ((high - low) / 2);
				if (X[mid] <= eval_pts[i]) {
					idx = mid + 1;
					low = mid + 1;
				} else if (X[mid] > eval_pts[i]) {
					high = mid;
				}
			}

		if (idx > 0)
			--idx;
		else if (idx > M - 2)
			idx = M - 2;

		spline_t const b_i =
			((Y[idx+1] - Y[idx]) / (X[idx+1] - X[idx])) -
			(((X[idx+1] - X[idx]) * (C[idx+1] + (2.0 * C[idx]))) / 3.0);
		spline_t const d_i = (C[idx+1] - C[idx]) / (3.0 * (X[idx+1] - X[idx]));
		values[i] = Y[idx] +
			b_i    * (eval_pts[i] - X[idx]) +
			C[idx] * (eval_pts[i] - X[idx]) * (eval_pts[i] - X[idx]) +
			d_i    * (eval_pts[i] - X[idx]) * (eval_pts[i] - X[idx]) * (eval_pts[i] - X[idx]);
	}
}

#define NFIT 121
#define N 3
int main(void) {
  spline_t psiNSpline[N] = {0.0, 0.6, 0.8};
  spline_t mHatSpline[N] = {0.0, 1.0, 2.0};
  spline_t eval[NFIT];
  for (size_t i = 0; i < NFIT; ++i)
    eval[i] = i * 0.01f;
  spline_t work[N];
  spline_t dummy[3*N];
  spline_t v[NFIT];
  struct spline s = { .n = N, .x = psiNSpline, .y = mHatSpline, .c = work };
  spline_eval(NFIT, v, eval, s, dummy);

  for (int i=0; i<N; i++)
    printf("%f\t",v[i]);
  printf("\n");
}
